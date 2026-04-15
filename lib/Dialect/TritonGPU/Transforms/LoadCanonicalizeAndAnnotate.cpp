#include "MeshAttrUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPULOADCANONICALIZEANDANNOTATE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool hasDotUser(Value value) {
  SmallVector<Value> worklist = {value};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      if (isa<triton::DotOp>(user))
        return true;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  return false;
}

static void collectProducerLoads(Value root,
                                 SmallVectorImpl<triton::LoadOp> &loads) {
  SmallVector<Value> worklist = {root};
  llvm::SmallPtrSet<Value, 32> visitedValues;
  llvm::SmallPtrSet<Operation *, 32> visitedOps;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visitedValues.insert(current).second)
      continue;
    auto definingOp = current.getDefiningOp();
    if (!definingOp)
      continue;

    if (auto loadOp = dyn_cast<triton::LoadOp>(definingOp)) {
      loads.push_back(loadOp);
      continue;
    }

    if (!visitedOps.insert(definingOp).second)
      continue;
    for (Value operand : definingOp->getOperands())
      worklist.push_back(operand);
  }
}

class TritonGPULoadCanonicalizeAndAnnotatePass
    : public impl::TritonGPULoadCanonicalizeAndAnnotateBase<
          TritonGPULoadCanonicalizeAndAnnotatePass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder b(mod.getContext());
    MeshDotAttrs dotAttrs;
    llvm::SmallPtrSet<Operation *, 64> seenLoads;

    mod.walk([&](triton::DotOp dotOp) {
      dotAttrs.annotateOp(dotOp, b, /*hasDotConsumer=*/true);

      SmallVector<triton::LoadOp> lhsLoads;
      SmallVector<triton::LoadOp> rhsLoads;
      collectProducerLoads(dotOp.getA(), lhsLoads);
      collectProducerLoads(dotOp.getB(), rhsLoads);

      for (triton::LoadOp loadOp : lhsLoads) {
        dotAttrs.annotateOp(loadOp, b, /*hasDotConsumer=*/true,
                            /*dotOperandIndex=*/0,
                            /*overwriteExisting=*/true);
        seenLoads.insert(loadOp.getOperation());
      }
      for (triton::LoadOp loadOp : rhsLoads) {
        dotAttrs.annotateOp(loadOp, b, /*hasDotConsumer=*/true,
                            /*dotOperandIndex=*/1,
                            /*overwriteExisting=*/true);
        seenLoads.insert(loadOp.getOperation());
      }
    });

    mod.walk([&](triton::LoadOp loadOp) {
      if (seenLoads.contains(loadOp.getOperation()))
        return;
      dotAttrs.annotateOp(loadOp, b, hasDotUser(loadOp.getResult()));
    });
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
