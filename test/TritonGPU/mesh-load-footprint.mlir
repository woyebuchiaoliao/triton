// RUN: triton-opt %s -split-input-file -tritongpu-load-canonicalize-and-annotate | FileCheck %s --check-prefix=ANNOTATE
// RUN: triton-opt %s -split-input-file -tritongpu-build-access-footprint | FileCheck %s --check-prefix=FOOTPRINT
// RUN: triton-opt %s -split-input-file -tritongpu-load-canonicalize-and-annotate -tritongpu-build-access-footprint -tritongpu-build-equivalence-class | FileCheck %s --check-prefix=EQ
// RUN: triton-opt %s -split-input-file -tritongpu-load-canonicalize-and-annotate -tritongpu-build-access-footprint -tritongpu-build-equivalence-class -tritongpu-optimize-load-layout-attr | FileCheck %s --check-prefix=OPT

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // ANNOTATE-LABEL: tt.func @mesh_annotate
  // ANNOTATE: %[[LA:.*]] = tt.load %{{.*arg0Off}}
  // ANNOTATE-SAME: mesh.load.flow = "west"
  // ANNOTATE: %[[LB:.*]] = tt.load %{{.*arg1Off}}
  // ANNOTATE-SAME: mesh.load.flow = "north"
  // ANNOTATE: tt.dot {{.*}} {mesh.compute.kind = "dot", mesh.dot.a.flow = "west", mesh.dot.b.flow = "north"}

  // FOOTPRINT-LABEL: tt.func @mesh_annotate
  // FOOTPRINT: %[[ZEROOFF:.*]] = arith.constant dense<0> : tensor<128x128xi32, #blocked>
  // FOOTPRINT: %[[OFFA:.*]] = tt.addptr %arg0, %[[ZEROOFF]]
  // FOOTPRINT: %[[LA2:.*]] = tt.load %[[OFFA]]
  // FOOTPRINT-SAME: mesh.fp.base_ptr = "%arg0"
  // FOOTPRINT-SAME: mesh.fp.mask.kind = "absent"
  // FOOTPRINT-SAME: mesh.fp.rank = 2 : i64
  // FOOTPRINT-SAME: mesh.fp.rw = "read"
  // FOOTPRINT-SAME: mesh.fp.shape = array<i64: 128, 128>
  // FOOTPRINT-SAME: mesh.fp.start_expr.slice =
  // FOOTPRINT-SAME: mesh.fp.stride = "unknown"
  //
  // EQ: %[[LA_EQ:.*]] = tt.load %{{.*arg0Off}}
  // EQ-SAME: mesh.domain.size = 32 : i64
  // EQ-SAME: mesh.eq.class.rule = "logical_block_id_plus_start_expr"
  // EQ-SAME: mesh.group.count = 1 : i64
  // EQ: %[[LA2_EQ:.*]] = tt.load %{{.*arg0Off}}
  // EQ-SAME: mesh.group.count = 1 : i64
  // EQ: %[[LB_EQ:.*]] = tt.load %{{.*arg1Off}}
  // EQ-SAME: mesh.group.count = 1 : i64
  //
  // OPT: %[[LA_OPT:.*]] = tt.load %{{.*arg0Off}}
  // OPT-SAME: mesh.opt.action = "broadcast_by_domain_group"
  // OPT-SAME: mesh.opt.reason = "domain_group_size_gt_1"
  // OPT-SAME: mesh.share.enable = true
  // OPT: %[[LA2_OPT:.*]] = tt.load %{{.*arg0Off}}
  // OPT-SAME: mesh.opt.action = "broadcast_by_domain_group"
  // OPT: %[[LB_OPT:.*]] = tt.load %{{.*arg1Off}}
  // OPT-SAME: mesh.opt.action = "broadcast_by_domain_group"

  tt.func @mesh_annotate(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked>,
                         %arg1: tensor<128x128x!tt.ptr<f16>, #blocked>) -> tensor<128x128xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %zeroOff = arith.constant dense<0> : tensor<128x128xi32, #blocked>
    %arg0Off = tt.addptr %arg0, %zeroOff : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %arg1Off = tt.addptr %arg1, %zeroOff : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %la = tt.load %arg0Off : tensor<128x128x!tt.ptr<f16>, #blocked>
    %la2 = tt.load %arg0Off : tensor<128x128x!tt.ptr<f16>, #blocked>
    %lb = tt.load %arg1Off : tensor<128x128x!tt.ptr<f16>, #blocked>
    %a = ttg.convert_layout %la : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %a2 = ttg.convert_layout %la2 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %b = ttg.convert_layout %lb : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %d = tt.dot %a, %b, %acc : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %d2 = tt.dot %a2, %b, %acc : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %sum = arith.addf %d, %d2 : tensor<128x128xf32, #mma>
    tt.return %sum : tensor<128x128xf32, #mma>
  }
}
