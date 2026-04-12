// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/rvv-rr2-p6-o.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Optimized by AutoComp (https://github.com/ucb-bar/autocomp)

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_o_u2v(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float xmin = -0x1.5ebb82p6;
  const float r_ln2f = 0x1.715476p+0f;
  const float l2uf = 0x1.62E400p-1f;
  const float l2lf = 0x1.7F7D1Cp-20f;
  const float c6 = 0x1.6850e4p-10f;
  const float c5 = 0x1.123bccp-7;
  const float c4 = 0x1.555b98p-5f;
  const float c3 = 0x1.55548ep-3f;
  const float c2 = 0x1.fffff8p-2f;

  const int32_t p = (24 - 1);
  const int32_t bias = (128 - 1);

  const float max_val = *max;

  size_t n = batch >> 2;
  size_t avl = n;
  size_t vl = __riscv_vsetvl_e32m2(avl);

  vfloat32m2_t vsum = __riscv_vfmv_v_f_f32m2(0.0f, vl);

  if (avl > 0) {
    vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(input, vl);
    input += vl;
    avl -= vl;

    do {
      vfloat32m2_t vx_proc = __riscv_vfsub_vf_f32m2(vx0, max_val, vl);
      vx_proc = __riscv_vfmax_vf_f32m2(vx_proc, xmin, vl);

      vfloat32m2_t vx1;
      size_t next_vl = 0;
      if (avl > 0) {
        next_vl = __riscv_vsetvl_e32m2(avl);
        vx1 = __riscv_vle32_v_f32m2(input, next_vl);
        input += next_vl;
      }

      vfloat32m2_t v = __riscv_vfmul_vf_f32m2(vx_proc, r_ln2f, vl);

      vint32m2_t q = __riscv_vfcvt_x_f_v_i32m2(v, vl);
      vfloat32m2_t z = __riscv_vfcvt_f_x_v_f32m2(q, vl);

      vfloat32m2_t s = __riscv_vfnmsac_vf_f32m2(vx_proc, l2uf, z, vl);
      s = __riscv_vfnmsac_vf_f32m2(s, l2lf, z, vl);

      vfloat32m2_t y = __riscv_vfmv_v_f_f32m2(c5, vl);
      y = __riscv_vfmacc_vf_f32m2(y, c6, s, vl);
      y = __riscv_vfmadd_vv_f32m2(y, s, __riscv_vfmv_v_f_f32m2(c4, vl), vl);
      y = __riscv_vfmadd_vv_f32m2(y, s, __riscv_vfmv_v_f_f32m2(c3, vl), vl);
      y = __riscv_vfmadd_vv_f32m2(y, s, __riscv_vfmv_v_f_f32m2(c2, vl), vl);
      vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
      y = __riscv_vfmadd_vv_f32m2(y, s, one, vl);
      y = __riscv_vfmadd_vv_f32m2(y, s, one, vl);

      vint32m2_t qw = __riscv_vadd_vx_i32m2(q, bias, vl);
      vint32m2_t qq = __riscv_vsll_vx_i32m2(qw, p, vl);
      vfloat32m2_t qf = __riscv_vreinterpret_v_i32m2_f32m2(qq);

      vfloat32m2_t vexp = __riscv_vfmul_vv_f32m2(y, qf, vl);

      __riscv_vse32_v_f32m2(output, vexp, vl);
      output += vl;
      vsum = __riscv_vfadd_vv_f32m2_tu(vsum, vsum, vexp, vl);

      vx0 = vx1;
      vl = next_vl;
      avl -= vl;
    } while (vl > 0);
  }

  vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  *sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vsum, v0, n));
}
