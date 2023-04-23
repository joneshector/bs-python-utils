import numpy as np
import pandas as pd

from bs_python_utils.pandas_utils import (
    _test_names_n,
    bspd_cross_products,
    bspd_prepareplot,
    bspd_print,
    bspd_statsdf,
)

df = pd.DataFrame({"a": [4, 2, 3], "b": [-1, 2, 0], "c": [1, 4, 2]})

l1 = ["a", "b", "c"]
l2 = ["c", "a"]

df_cprods1 = bspd_cross_products(df, l1, l2)
df_cprods2 = bspd_cross_products(df, l1, with_squares=False)
df_cprods3 = bspd_cross_products(df, l1)
df_cprods4 = bspd_cross_products(df, l1, l2, with_squares=False)

bspd_print(df, "original")
bspd_print(df_cprods1, "l1*l2 with squares")
bspd_print(df_cprods2, "l1*l1 without squares")
bspd_print(df_cprods3, "l1*l1 with squares")
bspd_print(df_cprods4, "l1*l2 without squares")

T1 = np.arange(12).reshape((4, 3))
T1_names = ["a", "b", "c"]
df_T1 = bspd_statsdf(T1, T1_names)
bspd_print(df_T1, "T1 matrix")

T2 = np.arange(8).reshape((4, 2))
T2_names = ["a", "d"]
df_T12 = bspd_statsdf([T1, T2], [T1_names, T2_names])
bspd_print(df_T12, "T1 with T2")

T3 = np.arange(8).reshape((4, 2))
T3_names = ["c", "e"]
df_T123 = bspd_statsdf([T1, T2, T3], [T1_names, T2_names, T3_names])
bspd_print(df_T123, "T1 with T2 and T3")

print(_test_names_n(["a"]))
print(_test_names_n(["a_true"]))
print(_test_names_n(["a_1"]))
print(_test_names_n(["a_1", "b_2"]))
#     next two should fail
# print(_test_names_n(['a_1', 'a_true']))
# print(_test_names_n(['a_1', 'b']))


bspd_print(bspd_prepareplot(df_T1))
bspd_print(bspd_prepareplot(df_T12))
bspd_print(bspd_prepareplot(df_T123))

# dfp = bspd_prepareplot(df_T123)
#
# import altair as alt
#
# ch = alt.Chart(dfp).mark_point().encode(
#         x='Sample:O',
#         y='Value:Q',
#         color=alt.Color('Group:N'),
#         facet=alt.Facet('Statistic:N')
#     ).resolve_scale(y='independent')
#
# ch.save("try.html")
