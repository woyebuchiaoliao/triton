#include "MeshAttrUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallDenseMap.h"
#include <algorithm>

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZELOADLAYOUTATTR
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class TritonGPUOptimizeLoadLayoutAttrPass
    : public impl::TritonGPUOptimizeLoadLayoutAttrBase<
          TritonGPUOptimizeLoadLayoutAttrPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder b(mod.getContext());

    mod.walk([&](triton::LoadOp loadOp) {
      auto groupMapAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
          loadOp->getAttr(MeshAttrKeys::groupMap));
      auto leadersAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
          loadOp->getAttr(MeshAttrKeys::groupLeaders));
      auto activeBlocksAttr = dyn_cast_or_null<IntegerAttr>(
          loadOp->getAttr(MeshAttrKeys::domainActiveBlocks));

      int64_t activeBlocks = activeBlocksAttr ? activeBlocksAttr.getInt() : 0;

      StringRef maskKind = "absent";
      if (auto maskKindAttr = dyn_cast_or_null<StringAttr>(
              loadOp->getAttr(MeshAttrKeys::fpMaskKind))) {
        maskKind = maskKindAttr.getValue();
      }

      // Missing domain-group metadata => safe fallback.
      if (!groupMapAttr || !leadersAttr || activeBlocks <= 0) {
        loadOp->setAttr(MeshAttrKeys::shareEnable, b.getBoolAttr(false));
        loadOp->setAttr(MeshAttrKeys::shareKind, b.getStringAttr("none"));
        loadOp->setAttr(MeshAttrKeys::optAction, b.getStringAttr("no_share"));
        loadOp->setAttr(MeshAttrKeys::optReason,
                        b.getStringAttr("missing_domain_group_info"));
        return;
      }

      // Compute max group size from per-lane group ids.
      SmallDenseMap<int64_t, int64_t> groupSizes;
      for (int64_t lane = 0, e = groupMapAttr.size(); lane < e; ++lane) {
        int64_t gid = groupMapAttr[lane];
        if (lane >= activeBlocks || gid < 0)
          continue;
        groupSizes[gid] += 1;
      }
      int64_t maxGroupSize = 0;
      for (auto &it : groupSizes)
        maxGroupSize = std::max(maxGroupSize, it.second);

      bool canBroadcast = maxGroupSize > 1 &&
                          (maskKind == "absent" || maskKind == "none");

      if (canBroadcast) {
        loadOp->setAttr(MeshAttrKeys::shareEnable, b.getBoolAttr(true));
        loadOp->setAttr(MeshAttrKeys::shareKind, b.getStringAttr("broadcast"));
        loadOp->setAttr(MeshAttrKeys::optAction,
                        b.getStringAttr("broadcast_by_domain_group"));
        loadOp->setAttr(MeshAttrKeys::optReason,
                        b.getStringAttr("domain_group_size_gt_1"));
        if (!leadersAttr.empty()) {
          loadOp->setAttr(MeshAttrKeys::optLeader,
                          b.getI64IntegerAttr(leadersAttr[0]));
        }
      } else {
        loadOp->setAttr(MeshAttrKeys::shareEnable, b.getBoolAttr(false));
        loadOp->setAttr(MeshAttrKeys::shareKind, b.getStringAttr("none"));
        loadOp->setAttr(MeshAttrKeys::optAction, b.getStringAttr("no_share"));
        loadOp->setAttr(MeshAttrKeys::optReason,
                        b.getStringAttr(maxGroupSize <= 1
                                            ? "all_domain_groups_singleton"
                                            : "masked_load"));
      }

      if (!loadOp->hasAttr(MeshAttrKeys::layoutOptimized)) {
        loadOp->setAttr(MeshAttrKeys::layoutOptimized,
                        b.getStringAttr("row_read_row"));
      }
    });
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
