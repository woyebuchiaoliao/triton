// RUN: triton-opt %s -split-input-file -tritongpu-build-access-footprint -tritongpu-build-equivalence-class | FileCheck %s

module {
  // CHECK-LABEL: tt.func @row_like_const_ub
  // CHECK: tt.load {{.*}}mesh.eq.class.rule = "logical_block_id_plus_start_expr"
  // CHECK-SAME: mesh.group.count = 4 : i64
  tt.func @row_like_const_ub(%base: !tt.ptr<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32

    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (f32) {
      %pid = tt.get_program_id axis = 0 : i32
      %c8 = arith.constant 8 : i32
      %off = arith.divsi %pid, %c8 : i32
      %ptr = tt.addptr %base, %off : !tt.ptr<f32>, i32
      %v = tt.load %ptr : !tt.ptr<f32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }
    tt.return %out : f32
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @matmul_like_masked_b_load
  // CHECK: tt.load {{.*}}mesh.eq.class.rule = "logical_block_id_plus_start_expr"
  // CHECK-SAME: mesh.group.count = 4 : i64
  tt.func @matmul_like_masked_b_load(%b_ptr: !tt.ptr<f16>, %K: i32, %stride_bk: i32) -> tensor<32x32xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.0> : tensor<32x32xf16>

    %pid = tt.get_program_id axis = 0 : i32
    %pid_off = arith.divsi %pid, %c8_i32 : i32

    %r32 = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %r32 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %r32 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>

    %pid_t = tt.splat %pid_off : i32 -> tensor<32x1xi32>
    %pid_bc = tt.broadcast %pid_t : tensor<32x1xi32> -> tensor<32x32xi32>
    %col_bc = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
    %base_off = arith.addi %pid_bc, %col_bc : tensor<32x32xi32>

    %b_ptrs = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
    %b_ptrs_27 = tt.addptr %b_ptrs, %base_off : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
    %stride_step = arith.muli %stride_bk, %c32_i32 : i32

    %res = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%acc = %cst) -> (tensor<32x32xf16>) : i32 {
      %koff = arith.muli %k, %stride_step : i32
      %koff_t = tt.splat %koff : i32 -> tensor<32x32xi32>
      %b_ptrs_50 = tt.addptr %b_ptrs_27, %koff_t : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>

      %k_rem = arith.subi %K, %koff : i32
      %k_rem_t = tt.splat %k_rem : i32 -> tensor<32x1xi32>
      %mask2d = arith.cmpi slt, %rows, %k_rem_t : tensor<32x1xi32>
      %mask = tt.broadcast %mask2d : tensor<32x1xi1> -> tensor<32x32xi1>

      %ld = tt.load %b_ptrs_50, %mask, %cst : tensor<32x32x!tt.ptr<f16>>
      scf.yield %ld : tensor<32x32xf16>
    }
    tt.return %res : tensor<32x32xf16>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @row_like_arg_ub
  // CHECK: tt.load {{.*}}mesh.eq.class.rule = "logical_block_id_plus_start_expr"
  // CHECK-SAME: mesh.group.count = 4 : i64
  tt.func @row_like_arg_ub(%base: !tt.ptr<f32>, %ub: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    %out = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %zero) -> (f32) {
      %pid = tt.get_program_id axis = 0 : i32
      %c8 = arith.constant 8 : i32
      %off = arith.divsi %pid, %c8 : i32
      %ptr = tt.addptr %base, %off : !tt.ptr<f32>, i32
      %v = tt.load %ptr : !tt.ptr<f32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }
    tt.return %out : f32
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @row_like_chained_addptr
  // CHECK: tt.load {{.*}}mesh.eq.class.rule = "logical_block_id_plus_start_expr"
  // CHECK-SAME: mesh.group.count = 4 : i64
  tt.func @row_like_chained_addptr(%base: !tt.ptr<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %zero = arith.constant 0.0 : f32

    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (f32) {
      %pid = tt.get_program_id axis = 0 : i32
      %c8 = arith.constant 8 : i32
      %off = arith.divsi %pid, %c8 : i32
      %ptr1 = tt.addptr %base, %off : !tt.ptr<f32>, i32
      %ptr2 = tt.addptr %ptr1, %c0_i32 : !tt.ptr<f32>, i32
      %v = tt.load %ptr2 : !tt.ptr<f32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }
    tt.return %out : f32
  }
}
