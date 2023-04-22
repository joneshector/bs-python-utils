""" a personal library of plots
"""

import altair as alt
from altair_saver import save as alt_save

from bs_python_utils.bs_altair import (
    alt_density,
    alt_faceted_densities,
    alt_histogram_by,
    alt_histogram_continuous,
    alt_lineplot,
    alt_linked_scatterplots,
    alt_scatterplot,
    alt_scatterplot_with_histo,
    alt_stacked_area,
    alt_stacked_area_facets,
    alt_superposed_faceted_lineplot,
    alt_superposed_lineplot,
    alt_tick_plots,
    plot_parameterized_estimates,
    plot_true_sim2_facets,
    plot_true_sim_facets,
)
from bs_seaborn import (
    bs_sns_bar_x_byf,
    bs_sns_bar_x_byfg,
    bs_sns_density_estimates,
    bs_sns_get_legend,
)
from bsmplutils import ax_text
