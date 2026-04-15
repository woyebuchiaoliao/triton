#include "MeshAttrUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUBUILDACCESSFOOTPRINT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static std::string valueToString(Value value, AsmState &asmState) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  value.printAsOperand(os, asmState);
  return os.str();
}

static StringAttr operationToStringAttr(Operation *op, Builder &b) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op->print(os, OpPrintingFlags().skipRegions());
  return b.getStringAttr(os.str());
}

static bool isPointerLikeType(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  return tensorTy && isa<triton::PointerType>(tensorTy.getElementType());
}

static Value traceBasePtrRoot(Value ptr) {
  Value current = ptr;
  llvm::SmallPtrSet<Value, 16> visited;
  while (visited.insert(current).second) {
    if (isa<BlockArgument>(current))
      return current;
    Operation *def = current.getDefiningOp();
    if (!def)
      return current;

    Value next;
    for (Value operand : def->getOperands()) {
      if (isPointerLikeType(operand.getType())) {
        next = operand;
        break;
      }
    }
    if (!next)
      return current;
    current = next;
  }
  return current;
}

class TritonGPUBuildAccessFootprintPass
    : public impl::TritonGPUBuildAccessFootprintBase<
          TritonGPUBuildAccessFootprintPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    AsmState asmState(mod);
    Builder b(mod.getContext());
    MeshAccessFootprintAttrs footprintAttrs;

    mod.walk([&](triton::LoadOp loadOp) {
      footprintAttrs.setReadWriteKind(loadOp, b);
      Value rootBasePtr = traceBasePtrRoot(loadOp.getPtr());
      footprintAttrs.setBasePtr(loadOp, b,
                                valueToString(rootBasePtr, asmState));

      SetVector<Operation *> ptrSlice;
      BackwardSliceOptions options;
      options.omitUsesFromAbove = false;
      (void)getBackwardSlice(loadOp.getPtr(), &ptrSlice, options);

      SmallVector<Attribute> ptrSliceAttrs;
      ptrSliceAttrs.reserve(ptrSlice.size());
      for (Operation *op : ptrSlice)
        ptrSliceAttrs.push_back(operationToStringAttr(op, b));
      footprintAttrs.setStartExprSlice(loadOp, b, ptrSliceAttrs);

      Type resultTy = loadOp.getType();
      if (auto tensorTy = dyn_cast<RankedTensorType>(resultTy))
        footprintAttrs.setShapeAndRank(loadOp, b, tensorTy);

      if (loadOp.getMask()) {
        footprintAttrs.setMask(loadOp, b,
                               valueToString(loadOp.getMask(), asmState));
      } else {
        footprintAttrs.setMask(loadOp, b, std::nullopt);
      }

      // TODO(mesh-pass2): infer symbolic stride from ptr expression for precise
      // equivalence checks.
      footprintAttrs.setUnknownStride(loadOp, b);
    });
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
