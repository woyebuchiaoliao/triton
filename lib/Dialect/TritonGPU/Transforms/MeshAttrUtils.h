#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_MESHATTRUTILS_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_MESHATTRUTILS_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>
#include <string>

namespace mlir {
namespace triton {
namespace gpu {

struct MeshAttrKeys {
  static constexpr StringLiteral loadKind = "mesh.load.kind";
  static constexpr StringLiteral eqClassRule = "mesh.eq.class.rule";
  static constexpr StringLiteral shareEnable = "mesh.share.enable";
  static constexpr StringLiteral shareKind = "mesh.share.kind";
  static constexpr StringLiteral layoutDefault = "mesh.layout.default";
  static constexpr StringLiteral layoutOptimized = "mesh.layout.optimized";
  static constexpr StringLiteral computeConsumerHint =
      "mesh.compute.consumer_hint";
  static constexpr StringLiteral loadFlow = "mesh.load.flow";

  static constexpr StringLiteral computeKind = "mesh.compute.kind";
  static constexpr StringLiteral dotAFlow = "mesh.dot.a.flow";
  static constexpr StringLiteral dotBFlow = "mesh.dot.b.flow";

  static constexpr StringLiteral fpRw = "mesh.fp.rw";
  static constexpr StringLiteral fpBasePtr = "mesh.fp.base_ptr";
  static constexpr StringLiteral fpStartExprSlice = "mesh.fp.start_expr.slice";
  static constexpr StringLiteral fpRank = "mesh.fp.rank";
  static constexpr StringLiteral fpShape = "mesh.fp.shape";
  static constexpr StringLiteral fpMaskKind = "mesh.fp.mask.kind";
  static constexpr StringLiteral fpMaskValue = "mesh.fp.mask.value";
  static constexpr StringLiteral fpStride = "mesh.fp.stride";

  static constexpr StringLiteral eqClassId = "mesh.eq.class.id";
  static constexpr StringLiteral eqClassSize = "mesh.eq.class.size";
  static constexpr StringLiteral domainSize = "mesh.domain.size";
  static constexpr StringLiteral domainRows = "mesh.domain.rows";
  static constexpr StringLiteral domainCols = "mesh.domain.cols";
  static constexpr StringLiteral domainActiveBlocks = "mesh.domain.active_blocks";
  static constexpr StringLiteral groupMap = "mesh.group.map";
  static constexpr StringLiteral groupCount = "mesh.group.count";
  static constexpr StringLiteral groupLeaders = "mesh.group.leaders";
  static constexpr StringLiteral optAction = "mesh.opt.action";
  static constexpr StringLiteral optReason = "mesh.opt.reason";
  static constexpr StringLiteral optLeader = "mesh.opt.leader";
};

struct MeshDotAttrs {
  StringRef loadKind = "unknown";
  StringRef eqClassRule = "none";
  bool shareEnable = false;
  StringRef shareKind = "none";
  StringRef layoutDefault = "row_read_row";
  StringRef layoutOptimized = "row_read_row";
  StringRef loadFlow = "unknown";
  StringRef computeKind = "dot";

  StringRef flowFromDotOperandIndex(unsigned operandIndex) const {
    if (operandIndex == 0)
      return "west";
    if (operandIndex == 1)
      return "north";
    return "unknown";
  }

  void annotateLoad(triton::LoadOp loadOp, Builder &builder, bool hasDotConsumer,
                    std::optional<unsigned> dotOperandIndex = std::nullopt,
                    bool overwriteExisting = false) const {
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::loadKind))
      loadOp->setAttr(MeshAttrKeys::loadKind, builder.getStringAttr(loadKind));
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::eqClassRule))
      loadOp->setAttr(MeshAttrKeys::eqClassRule,
                      builder.getStringAttr(eqClassRule));
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::shareEnable))
      loadOp->setAttr(MeshAttrKeys::shareEnable,
                      builder.getBoolAttr(shareEnable));
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::shareKind))
      loadOp->setAttr(MeshAttrKeys::shareKind,
                      builder.getStringAttr(shareKind));
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::layoutDefault))
      loadOp->setAttr(MeshAttrKeys::layoutDefault,
                      builder.getStringAttr(layoutDefault));
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::layoutOptimized))
      loadOp->setAttr(MeshAttrKeys::layoutOptimized,
                      builder.getStringAttr(layoutOptimized));
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::computeConsumerHint)) {
      loadOp->setAttr(MeshAttrKeys::computeConsumerHint,
                      builder.getStringAttr(hasDotConsumer ? "dot"
                                                           : "generic"));
    }
    StringRef flow = dotOperandIndex.has_value()
                         ? flowFromDotOperandIndex(*dotOperandIndex)
                         : loadFlow;
    if (overwriteExisting || !loadOp->hasAttr(MeshAttrKeys::loadFlow))
      loadOp->setAttr(MeshAttrKeys::loadFlow, builder.getStringAttr(flow));
  }

  void annotateDot(triton::DotOp dotOp, Builder &builder,
                   bool overwriteExisting = false) const {
    if (overwriteExisting || !dotOp->hasAttr(MeshAttrKeys::computeKind))
      dotOp->setAttr(MeshAttrKeys::computeKind,
                     builder.getStringAttr(computeKind));
    if (overwriteExisting || !dotOp->hasAttr(MeshAttrKeys::dotAFlow)) {
      dotOp->setAttr(MeshAttrKeys::dotAFlow,
                     builder.getStringAttr(flowFromDotOperandIndex(0)));
    }
    if (overwriteExisting || !dotOp->hasAttr(MeshAttrKeys::dotBFlow)) {
      dotOp->setAttr(MeshAttrKeys::dotBFlow,
                     builder.getStringAttr(flowFromDotOperandIndex(1)));
    }
  }

  void annotateOp(Operation *op, Builder &builder, bool hasDotConsumer,
                  std::optional<unsigned> dotOperandIndex = std::nullopt,
                  bool overwriteExisting = false) const {
    if (auto dotOp = dyn_cast<triton::DotOp>(op))
      return annotateDot(dotOp, builder, overwriteExisting);
    if (auto loadOp = dyn_cast<triton::LoadOp>(op))
      return annotateLoad(loadOp, builder, hasDotConsumer, dotOperandIndex,
                          overwriteExisting);
  }
};

struct MeshAccessFootprintAttrs {
  StringRef rw = "read";
  StringRef maskAbsent = "absent";
  StringRef maskPresent = "present";
  StringRef unknownStride = "unknown";

  void setReadWriteKind(triton::LoadOp loadOp, Builder &builder) const {
    loadOp->setAttr(MeshAttrKeys::fpRw, builder.getStringAttr(rw));
  }

  void setBasePtr(triton::LoadOp loadOp, Builder &builder,
                  const std::string &basePtr) const {
    loadOp->setAttr(MeshAttrKeys::fpBasePtr, builder.getStringAttr(basePtr));
  }

  void setStartExprSlice(triton::LoadOp loadOp, Builder &builder,
                         ArrayRef<Attribute> sliceAttrs) const {
    loadOp->setAttr(MeshAttrKeys::fpStartExprSlice,
                    builder.getArrayAttr(sliceAttrs));
  }

  void setShapeAndRank(triton::LoadOp loadOp, Builder &builder,
                       RankedTensorType tensorTy) const {
    loadOp->setAttr(MeshAttrKeys::fpRank,
                    builder.getI64IntegerAttr(tensorTy.getRank()));
    SmallVector<int64_t> shape;
    shape.reserve(tensorTy.getRank());
    for (int64_t dim : tensorTy.getShape())
      shape.push_back(dim);
    loadOp->setAttr(MeshAttrKeys::fpShape, builder.getDenseI64ArrayAttr(shape));
  }

  void setMask(triton::LoadOp loadOp, Builder &builder,
               std::optional<std::string> maskValue) const {
    if (!maskValue.has_value()) {
      loadOp->setAttr(MeshAttrKeys::fpMaskKind, builder.getStringAttr(maskAbsent));
      return;
    }
    loadOp->setAttr(MeshAttrKeys::fpMaskKind, builder.getStringAttr(maskPresent));
    loadOp->setAttr(MeshAttrKeys::fpMaskValue,
                    builder.getStringAttr(maskValue.value()));
  }

  void setUnknownStride(triton::LoadOp loadOp, Builder &builder) const {
    loadOp->setAttr(MeshAttrKeys::fpStride,
                    builder.getStringAttr(unknownStride));
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif
