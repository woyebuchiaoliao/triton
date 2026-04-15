#include "MeshAttrUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUBUILDEQUIVALENCECLASS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isPointerLikeType(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  return tensorTy && isa<triton::PointerType>(tensorTy.getElementType());
}

static FailureOr<int64_t> evaluateExpr(Value value, Value pid,
                                       int64_t pidValue,
                                       DenseMap<Value, FailureOr<int64_t>> &cache) {
  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;

  auto fail = [&]() -> FailureOr<int64_t> {
    FailureOr<int64_t> f = failure();
    cache[value] = f;
    return f;
  };

  if (value == pid) {
    FailureOr<int64_t> r(pidValue);
    cache[value] = r;
    return r;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (isPointerLikeType(blockArg.getType())) {
      FailureOr<int64_t> r(0);
      cache[value] = r;
      return r;
    }
    return fail();
  }

  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>()) {
    FailureOr<int64_t> r(cst.value());
    cache[value] = r;
    return r;
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>()) {
    FailureOr<int64_t> r(cst.value());
    cache[value] = r;
    return r;
  }
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    Attribute val = cst.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(val)) {
      FailureOr<int64_t> r(intAttr.getInt());
      cache[value] = r;
      return r;
    }
    if (auto dense = dyn_cast<DenseIntElementsAttr>(val)) {
      if (!dense.empty()) {
        FailureOr<int64_t> r((*dense.value_begin<APInt>()).getSExtValue());
        cache[value] = r;
        return r;
      }
    }
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return fail();

  auto evalBinary = [&](Value lhs, Value rhs,
                        function_ref<int64_t(int64_t, int64_t)> fn)
      -> FailureOr<int64_t> {
    FailureOr<int64_t> lhsV = evaluateExpr(lhs, pid, pidValue, cache);
    FailureOr<int64_t> rhsV = evaluateExpr(rhs, pid, pidValue, cache);
    if (failed(lhsV) || failed(rhsV))
      return fail();
    FailureOr<int64_t> r(fn(*lhsV, *rhsV));
    cache[value] = r;
    return r;
  };

  if (auto add = dyn_cast<arith::AddIOp>(def))
    return evalBinary(add.getLhs(), add.getRhs(),
                      [](int64_t a, int64_t b) { return a + b; });
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return evalBinary(sub.getLhs(), sub.getRhs(),
                      [](int64_t a, int64_t b) { return a - b; });
  if (auto mul = dyn_cast<arith::MulIOp>(def))
    return evalBinary(mul.getLhs(), mul.getRhs(),
                      [](int64_t a, int64_t b) { return a * b; });
  if (auto div = dyn_cast<arith::DivSIOp>(def))
    return evalBinary(div.getLhs(), div.getRhs(),
                      [](int64_t a, int64_t b) { return b == 0 ? 0 : a / b; });
  if (auto rem = dyn_cast<arith::RemSIOp>(def))
    return evalBinary(rem.getLhs(), rem.getRhs(),
                      [](int64_t a, int64_t b) { return b == 0 ? 0 : a % b; });
  if (auto cast = dyn_cast<arith::IndexCastOp>(def)) {
    FailureOr<int64_t> r = evaluateExpr(cast.getIn(), pid, pidValue, cache);
    cache[value] = r;
    return r;
  }
  if (auto splat = dyn_cast<triton::SplatOp>(def)) {
    FailureOr<int64_t> r = evaluateExpr(splat.getSrc(), pid, pidValue, cache);
    cache[value] = r;
    return r;
  }
  if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(def)) {
    FailureOr<int64_t> r = evaluateExpr(expandDims.getSrc(), pid, pidValue, cache);
    cache[value] = r;
    return r;
  }
  if (auto broadcast = dyn_cast<triton::BroadcastOp>(def)) {
    FailureOr<int64_t> r = evaluateExpr(broadcast.getSrc(), pid, pidValue, cache);
    cache[value] = r;
    return r;
  }
  if (auto makeRange = dyn_cast<triton::MakeRangeOp>(def)) {
    FailureOr<int64_t> r(makeRange.getStart());
    cache[value] = r;
    return r;
  }
  if (auto addPtr = dyn_cast<triton::AddPtrOp>(def))
    return evalBinary(addPtr.getPtr(), addPtr.getOffset(),
                      [](int64_t a, int64_t b) { return a + b; });

  return fail();
}

static Value findProgramId(Value startValue) {
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitUsesFromAbove = false;
  (void)getBackwardSlice(startValue, &slice, options);
  for (Operation *op : slice) {
    if (auto getPid = dyn_cast<triton::GetProgramIdOp>(op)) {
      if (getPid.getAxis() == 0)
        return getPid.getResult();
    }
  }
  return Value();
}

static Value getStartExprSeed(triton::LoadOp loadOp) {
  return loadOp->getNumOperands() > 0 ? loadOp->getOperand(0) : loadOp.getPtr();
}

class TritonGPUBuildEquivalenceClassPass
    : public impl::TritonGPUBuildEquivalenceClassBase<
          TritonGPUBuildEquivalenceClassPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    AsmState asmState(mod);
    Builder b(mod.getContext());

    const int64_t domainSize = 32;

    mod.walk([&](triton::LoadOp loadOp) {
      Value ptrOperand = loadOp->getNumOperands() > 0 ? loadOp->getOperand(0)
                                                      : loadOp.getPtr();
      std::string ptrKey;
      {
        llvm::raw_string_ostream os(ptrKey);
        ptrOperand.printAsOperand(os, asmState);
      }

      int64_t activeBlocks = domainSize;
      if (auto activeAttr = dyn_cast_or_null<IntegerAttr>(
              loadOp->getAttr(MeshAttrKeys::domainActiveBlocks))) {
        activeBlocks = std::min<int64_t>(domainSize,
                                         std::max<int64_t>(0, activeAttr.getInt()));
      }

      Value startSeed = getStartExprSeed(loadOp);
      Value pid = startSeed ? findProgramId(startSeed) : Value();

      llvm::StringMap<int64_t> signatureToGroup;
      SmallVector<int64_t> groupMap(domainSize, -1);
      SmallVector<int64_t> leaders;

      auto getGroupId = [&](StringRef signature, int64_t lane) {
        auto [it, inserted] = signatureToGroup.try_emplace(
            signature.str(), static_cast<int64_t>(signatureToGroup.size()));
        if (inserted)
          leaders.push_back(lane);
        return it->second;
      };

      for (int64_t lane = 0; lane < activeBlocks; ++lane) {
        std::string signature;
        if (startSeed && pid) {
          DenseMap<Value, FailureOr<int64_t>> cache;
          FailureOr<int64_t> v = evaluateExpr(startSeed, pid, lane, cache);
          if (succeeded(v)) {
            signature = (llvm::Twine("expr=") + llvm::Twine(*v)).str();
          } else {
            signature = (llvm::Twine("ptr=") + llvm::Twine(ptrKey)).str();
          }
        } else {
          // For now only the first load pointer operand participates in pass-3
          // grouping. Mask/other operands are intentionally ignored.
          signature = (llvm::Twine("ptr=") + llvm::Twine(ptrKey)).str();
        }
        groupMap[lane] = getGroupId(signature, lane);
      }

      int64_t groupCount = static_cast<int64_t>(signatureToGroup.size());

      loadOp->setAttr(MeshAttrKeys::eqClassRule,
                      b.getStringAttr("logical_block_id_plus_start_expr"));
      loadOp->setAttr(MeshAttrKeys::domainSize, b.getI64IntegerAttr(domainSize));
      loadOp->setAttr(MeshAttrKeys::domainActiveBlocks,
                      b.getI64IntegerAttr(activeBlocks));
      loadOp->setAttr(MeshAttrKeys::groupMap, b.getDenseI64ArrayAttr(groupMap));
      loadOp->setAttr(MeshAttrKeys::groupCount, b.getI64IntegerAttr(groupCount));
      loadOp->setAttr(MeshAttrKeys::groupLeaders,
                      b.getDenseI64ArrayAttr(leaders));

      // Compatibility attrs kept for downstream/backward compatibility.
      loadOp->setAttr(MeshAttrKeys::eqClassId, b.getI64IntegerAttr(0));
      loadOp->setAttr(MeshAttrKeys::eqClassSize,
                      b.getI64IntegerAttr(std::max<int64_t>(1, groupCount)));
    });
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
