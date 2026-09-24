Minimize
  2 x__t8
Subject To
  initial_residual_0: -1 u__t0 - 1 x__t0 + 8 initial_der(x) = 0
  initial_residual_1: -1 x__t0 + 8 initial_der(w) = 0
  initial_residual_2: 1 x__t0 + 1 y__t0 = 3
  initial_residual_3: 1 x__t0 = 1.1
  collocation_eq0_t0: -1 u__t1 - 8 x__t0 + 7 x__t1 = 0
  collocation_eq1_t0: -1 x__t1 - 8 w__t0 + 8 w__t1 = 0
  collocation_eq2_t0: 1 x__t1 + 1 y__t1 = 3
  collocation_eq0_t1: -1 u__t2 - 8 x__t1 + 7 x__t2 = 0
  collocation_eq1_t1: -1 x__t2 - 8 w__t1 + 8 w__t2 = 0
  collocation_eq2_t1: 1 x__t2 + 1 y__t2 = 3
  collocation_eq0_t2: -1 u__t3 - 8 x__t2 + 7 x__t3 = 0
  collocation_eq1_t2: -1 x__t3 - 8 w__t2 + 8 w__t3 = 0
  collocation_eq2_t2: 1 x__t3 + 1 y__t3 = 3
  collocation_eq0_t3: -1 u__t4 - 8 x__t3 + 7 x__t4 = 0
  collocation_eq1_t3: -1 x__t4 - 8 w__t3 + 8 w__t4 = 0
  collocation_eq2_t3: 1 x__t4 + 1 y__t4 = 3
  collocation_eq0_t4: -1 u__t5 - 8 x__t4 + 7 x__t5 = 0
  collocation_eq1_t4: -1 x__t5 - 8 w__t4 + 8 w__t5 = 0
  collocation_eq2_t4: 1 x__t5 + 1 y__t5 = 3
  collocation_eq0_t5: -1 u__t6 - 8 x__t5 + 7 x__t6 = 0
  collocation_eq1_t5: -1 x__t6 - 8 w__t5 + 8 w__t6 = 0
  collocation_eq2_t5: 1 x__t6 + 1 y__t6 = 3
  collocation_eq0_t6: -1 u__t7 - 8 x__t6 + 7 x__t7 = 0
  collocation_eq1_t6: -1 x__t7 - 8 w__t6 + 8 w__t7 = 0
  collocation_eq2_t6: 1 x__t7 + 1 y__t7 = 3
  collocation_eq0_t7: -1 u__t8 - 8 x__t7 + 7 x__t8 = 0
  collocation_eq1_t7: -1 x__t8 - 8 w__t7 + 8 w__t8 = 0
  collocation_eq2_t7: 1 x__t8 + 1 y__t8 = 3
  delay__pymoca_delay_0_1,1__t0: -1 x__t0 + 1 _pymoca_delay_0_I1,1I___t0 = 0
  delay__pymoca_delay_0_1,1__t1: -0.8 x__t0 - 0.2 x__t1 + 1 _pymoca_delay_0_I1,1I___t1 = 0
  delay__pymoca_delay_0_1,1__t2: -0.8 x__t1 - 0.2 x__t2 + 1 _pymoca_delay_0_I1,1I___t2 = 0
  delay__pymoca_delay_0_1,1__t3: -0.8 x__t2 - 0.2 x__t3 + 1 _pymoca_delay_0_I1,1I___t3 = 0
  delay__pymoca_delay_0_1,1__t4: -0.8 x__t3 - 0.2 x__t4 + 1 _pymoca_delay_0_I1,1I___t4 = 0
  delay__pymoca_delay_0_1,1__t5: -0.8 x__t4 - 0.2 x__t5 + 1 _pymoca_delay_0_I1,1I___t5 = 0
  delay__pymoca_delay_0_1,1__t6: -0.8 x__t5 - 0.2 x__t6 + 1 _pymoca_delay_0_I1,1I___t6 = 0
  delay__pymoca_delay_0_1,1__t7: -0.8 x__t6 - 0.2 x__t7 + 1 _pymoca_delay_0_I1,1I___t7 = 0
  delay__pymoca_delay_0_1,1__t8: -0.8 x__t7 - 0.2 x__t8 + 1 _pymoca_delay_0_I1,1I___t8 = 0
  delay__pymoca_delay_1_1,1__t0: -1 x__t0 + 1 _pymoca_delay_1_I1,1I___t0 = 0
  delay__pymoca_delay_1_1,1__t1: -1 x__t0 + 1 _pymoca_delay_1_I1,1I___t1 = 0
  delay__pymoca_delay_1_1,1__t2: -1 x__t1 + 1 _pymoca_delay_1_I1,1I___t2 = 0
  delay__pymoca_delay_1_1,1__t3: -1 x__t2 + 1 _pymoca_delay_1_I1,1I___t3 = 0
  delay__pymoca_delay_1_1,1__t4: -1 x__t3 + 1 _pymoca_delay_1_I1,1I___t4 = 0
  delay__pymoca_delay_1_1,1__t5: -1 x__t4 + 1 _pymoca_delay_1_I1,1I___t5 = 0
  delay__pymoca_delay_1_1,1__t6: -1 x__t5 + 1 _pymoca_delay_1_I1,1I___t6 = 0
  delay__pymoca_delay_1_1,1__t7: -1 x__t6 + 1 _pymoca_delay_1_I1,1I___t7 = 0
  delay__pymoca_delay_1_1,1__t8: -1 x__t7 + 1 _pymoca_delay_1_I1,1I___t8 = 0
Bounds
  -2 <= u__t0 <= 2
  -2 <= u__t1 <= 2
  -2 <= u__t2 <= 2
  -2 <= u__t3 <= 2
  -2 <= u__t4 <= 2
  -2 <= u__t5 <= 2
  -2 <= u__t6 <= 2
  -2 <= u__t7 <= 2
  -2 <= u__t8 <= 2
  x__t0 Free
  x__t1 Free
  x__t2 Free
  x__t3 Free
  x__t4 Free
  x__t5 Free
  x__t6 Free
  x__t7 Free
  x__t8 Free
  0 <= w__t0 <= 0
  w__t1 Free
  w__t2 Free
  w__t3 Free
  w__t4 Free
  w__t5 Free
  w__t6 Free
  w__t7 Free
  w__t8 Free
  y__t0 Free
  y__t1 Free
  y__t2 Free
  y__t3 Free
  y__t4 Free
  y__t5 Free
  y__t6 Free
  y__t7 Free
  y__t8 Free
  _pymoca_delay_0_I1,1I___t0 Free
  _pymoca_delay_0_I1,1I___t1 Free
  _pymoca_delay_0_I1,1I___t2 Free
  _pymoca_delay_0_I1,1I___t3 Free
  _pymoca_delay_0_I1,1I___t4 Free
  _pymoca_delay_0_I1,1I___t5 Free
  _pymoca_delay_0_I1,1I___t6 Free
  _pymoca_delay_0_I1,1I___t7 Free
  _pymoca_delay_0_I1,1I___t8 Free
  _pymoca_delay_1_I1,1I___t0 Free
  _pymoca_delay_1_I1,1I___t1 Free
  _pymoca_delay_1_I1,1I___t2 Free
  _pymoca_delay_1_I1,1I___t3 Free
  _pymoca_delay_1_I1,1I___t4 Free
  _pymoca_delay_1_I1,1I___t5 Free
  _pymoca_delay_1_I1,1I___t6 Free
  _pymoca_delay_1_I1,1I___t7 Free
  _pymoca_delay_1_I1,1I___t8 Free
  initial_der(x) Free
  initial_der(w) Free
General
u__t0
u__t1
u__t2
u__t3
u__t4
u__t5
u__t6
u__t7
u__t8
End