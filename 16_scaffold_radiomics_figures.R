#!/usr/bin/env Rscript
# Radiomic comparison of Dr. Ren's scaffold materials: CG vs MC vs PEG-MC.
#
# Uses the SAME pre-specified 21-feature panel, in the same five themes, as the
# rabbit Defect vs Soaked-OPG vs C-OPG figures (9_opg_figures.py: HEAT_COMMON /
# IV_FAMILIES), so both studies present the same features. Nothing is selected
# by p-value; every panel feature is shown.
#
# Adaptations forced by the specimen (no bone, no reference ring, free-form region):
#   * "BV/TV" / "bone" -> dense fraction / dense phase (> 226 HU: mineral particles, gold);
#   * no defect : reference ratio — a scaffold has no surrounding ring;
#   * three features scale with region size BY DEFINITION and CG scaffolds are smaller
#     (about 7.5 mm vs 8.1–8.3 mm across), so they are expressed per unit volume:
#       component count -> components per mm3; NGTDM coarseness -> x voxels;
#       NGTDM busyness -> per 1,000 voxels.
#     (In the rabbit study the ROI is a fixed template, so this never arises there.)
#     2026-11-17: an earlier version of this script ranked all 111 features by p-value;
#     its top features (GLRLM / GLSZM / GLDM non-uniformity, NGTDM coarseness) tracked
#     region volume (rho 0.88–0.99) and were withdrawn.
#
# Input : Dr. Ren/results/radiomics/scaffold_radiomics.csv  (15_scaffold_radiomics.py)
# Output: Dr. Ren/results/radiomics/figures/
#   RadFig1_heatmap_<region>   21 features x scans, rows split CG / MC / PEG-MC
#   RadFig2_panel_<region>     every panel feature as a boxplot
#   RadFig3_summary            effect size of each feature in both regions + PCA
#   RadFig4_batch_correction_<region>  PCA before / after ComBat, variance by run vs material, pairwise PERMANOVA
#   RadFig1_heatmap_<region>_batchcorrected  the heatmap on ComBat-corrected features
#   radiomics_batch_permanova.csv, ../scaffold_radiomics_panel_combat_<region>.csv
#   radiomics_panel_tests.csv  the 21 features x 2 regions: medians, tests, effect sizes
#   radiomics_all_feature_tests.csv  all extracted features with a size-dependence flag
#   figure_legends.md
# Regions: interior (> 0.25 mm below the seeded surface; material without the cell
# crust — the primary region) and whole (entire scaffold VOI).
#
# Statistics (labelled scaffolds, runs R2–R4; duplicate rescan excluded):
#   Kruskal–Wallis across materials, epsilon2 effect size, Benjamini–Hochberg FDR over
#   the 21 panel features (q); run-adjusted type-II ANOVA of the rank-normalised
#   feature ~ scaffold + run (BH); "robust" = q < 0.05 in both; pairwise Wilcoxon, Holm.
# Stack: ggplot2, ggpubr/rstatix, ggbeeswarm, patchwork, pheatmap, ragg/cairo.
#
#   Rscript 16_scaffold_radiomics_figures.R

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(tibble); library(purrr)
  library(ggpubr); library(rstatix); library(patchwork); library(pheatmap); library(ggbeeswarm); library(grid)
})
options(warn = 1, dplyr.summarise.inform = FALSE)

args  <- commandArgs(trailingOnly = FALSE)
fpath <- sub("--file=", "", args[grepl("^--file=", args)])
REPO  <- if (length(fpath)) normalizePath(dirname(fpath)) else getwd()
ROOT  <- normalizePath(file.path(REPO, "..", "..", "..", "Dr. Ren"))
IN    <- file.path(ROOT, "results", "radiomics", "scaffold_radiomics.csv")
OUT   <- file.path(ROOT, "results", "radiomics", "figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── palette & theme ───────────────────────────────────────────────────────────
PAL   <- c(CG = "#2a78d6", MC = "#eb6834", `PEG-MC` = "#1baf7a")
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; LIGHT <- "#c3c2b7"
SHP   <- c(R2 = 16, R3 = 17, R4 = 15)
THEMES <- c("Dense material amount", "Dense-phase density", "Dense-phase structure",
            "Intensity distribution", "Texture")
THEME_COL <- setNames(c("#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"), THEMES)
FONT  <- "Arial"
theme_set(theme_pubr(base_size = 8, base_family = FONT, legend = "right") +
  theme(axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3),
        axis.text = element_text(colour = INK2), axis.title = element_text(colour = INK),
        plot.tag = element_text(face = "bold", size = 10),
        plot.subtitle = element_text(size = 7, colour = INK2),
        strip.background = element_blank(), strip.text = element_text(face = "bold", size = 7.5),
        legend.key.size = unit(3.5, "mm"), legend.text = element_text(size = 7),
        legend.title = element_text(size = 7.5)))
p_fmt <- function(p) ifelse(is.na(p), "n/a", ifelse(p < 0.001, "< 0.001", ifelse(p < 0.01, sprintf("= %.3f", p), sprintf("= %.2f", p))))
save_fig <- function(p, name, w, h) {
  ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = w, height = h, units = "mm", device = cairo_pdf)
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = w, height = h, units = "mm", dpi = 600,
         device = ragg::agg_png, bg = "white")
  cat("  wrote", name, "\n")
}
rank_norm <- function(x) { r <- rank(x, na.last = "keep"); qnorm((r - 0.5) / sum(!is.na(x))) }

# ── the feature panel (same 21 features and themes as 9_opg_figures.py) ───────
PANEL <- tribble(
  ~theme,                   ~key,                        ~label,                               ~short,
  "Dense material amount",  "dense_pct",                 "Dense fraction, > 226 HU (%)",       "Dense fraction",
  "Dense material amount",  "mean_hu",                   "Mean intensity (HU)",                "Mean HU",
  "Dense material amount",  "p90_hu",                    "90th percentile (HU)",               "P90 HU",
  "Dense-phase density",    "bone_mean_hu",              "Dense-phase mean intensity (HU)",    "Dense-phase mean HU",
  "Dense-phase structure",  "comp_density",              "Dense components (per mm³)",         "Components / mm³",
  "Dense-phase structure",  "bone_volume_to_surface_mm", "Dense-phase volume / surface (mm)",  "Volume / surface",
  "Dense-phase structure",  "bone_sphericity",           "Dense-phase sphericity",             "Sphericity",
  "Dense-phase structure",  "bone_solidity",             "Dense-phase solidity",               "Solidity",
  "Intensity distribution", "std_hu",                    "Intensity SD (HU)",                  "SD HU",
  "Intensity distribution", "skewness",                  "Skewness",                           "Skewness",
  "Intensity distribution", "kurtosis",                  "Kurtosis",                           "Kurtosis",
  "Intensity distribution", "entropy_bits",              "Histogram entropy (bits)",           "Entropy",
  "Texture",                "glcm_contrast",             "GLCM contrast",                      "GLCM contrast",
  "Texture",                "glcm_homogeneity",          "GLCM homogeneity",                   "GLCM homogeneity",
  "Texture",                "glcm_correlation",          "GLCM correlation",                   "GLCM correlation",
  "Texture",                "glcm_entropy_bits",         "GLCM entropy (bits)",                "GLCM entropy",
  "Texture",                "run_pct",                   "GLRLM run percentage (%)",           "GLRLM run %",
  "Texture",                "zone_pct",                  "GLSZM zone percentage (%)",          "GLSZM zone %",
  "Texture",                "glszm_large_area_emphasis", "GLSZM large-area emphasis",          "GLSZM large area",
  "Texture",                "coarse_n",                  "NGTDM coarseness (size-normalised)", "NGTDM coarseness*",
  "Texture",                "busy_n",                    "NGTDM busyness (per 1,000 voxels)",  "NGTDM busyness*"
) %>% mutate(theme = factor(theme, levels = THEMES), key = factor(key, levels = key))
KEYS <- as.character(PANEL$key)

# ── data ──────────────────────────────────────────────────────────────────────
raw <- read_csv(IN, col_types = cols(scan = col_character(), run = col_character(), batch = col_character(),
                                     scaffold = col_character(), labeled = col_character(), note = col_character(),
                                     .default = col_double()), progress = FALSE) %>%
  filter(!grepl("duplicate", coalesce(note, "")), run %in% names(SHP)) %>%
  mutate(control = labeled %in% c("none", "unlabeled"),
         scaffold = factor(scaffold, levels = names(PAL)), run = factor(run, levels = names(SHP)))

region_data <- function(rg) {
  g <- function(f) raw[[paste0(rg, "_", f)]]
  out <- tibble(scan = raw$scan, scaffold = raw$scaffold, run = raw$run, control = raw$control,
                volume_mm3 = g("volume_mm3"),
                dense_pct = 100 * g("bvtv_fixed"), mean_hu = g("mean_hu"), p90_hu = g("p90_hu"),
                bone_mean_hu = g("bone_mean_hu"),
                comp_density = g("bone_components") / g("volume_mm3"),
                bone_volume_to_surface_mm = g("bone_volume_to_surface_mm"),
                bone_sphericity = g("bone_sphericity"), bone_solidity = g("bone_solidity"),
                std_hu = g("std_hu"), skewness = g("skewness"), kurtosis = g("kurtosis"), entropy_bits = g("entropy_bits"),
                glcm_contrast = g("glcm_contrast"), glcm_homogeneity = g("glcm_homogeneity"),
                glcm_correlation = g("glcm_correlation"), glcm_entropy_bits = g("glcm_entropy_bits"),
                run_pct = 100 * g("glrlm_run_percentage"), zone_pct = 100 * g("glszm_zone_percentage"),
                glszm_large_area_emphasis = g("glszm_large_area_emphasis"),
                coarse_n = g("ngtdm_coarseness") * g("voxels"),
                busy_n = 1000 * g("ngtdm_busyness") / g("voxels"))
  stopifnot(all(KEYS %in% names(out)))
  out
}
REG <- list(interior = region_data("interior"), whole = region_data("whole"))
lab_of  <- function(rg) filter(REG[[rg]], !control)
ctrl_of <- function(rg) filter(REG[[rg]], control)
n_lab <- table(lab_of("interior")$scaffold)
cat(sprintf("labelled R2-R4: %d scans (%s); controls: %d; panel: %d features\n", nrow(lab_of("interior")),
            paste(names(n_lab), n_lab, sep = "=", collapse = ", "), nrow(ctrl_of("interior")), length(KEYS)))

# ── tests ─────────────────────────────────────────────────────────────────────
test_one <- function(y, scaffold, run) {
  ok <- is.finite(y)
  df <- tibble(y = y[ok], scaffold = scaffold[ok], run = run[ok])
  kw <- kruskal.test(y ~ scaffold, data = df)
  df$z <- rank_norm(df$y)
  av <- tryCatch(car::Anova(lm(z ~ scaffold + run, data = df), type = 2), error = function(e) NULL)
  pw <- suppressWarnings(pairwise.wilcox.test(df$y, df$scaffold, p.adjust.method = "holm", exact = FALSE))$p.value
  med <- tapply(df$y, df$scaffold, median)
  tibble(n = nrow(df), median_CG = med[["CG"]], median_MC = med[["MC"]], median_PEGMC = med[["PEG-MC"]],
         kw_H = unname(kw$statistic), kw_p = kw$p.value, epsilon2 = unname(kw$statistic) / (nrow(df) - 1),
         run_adj_p = if (is.null(av)) NA else av["scaffold", "Pr(>F)"],
         run_p = if (is.null(av)) NA else av["run", "Pr(>F)"],
         holm_CG_MC = pw["MC", "CG"], holm_CG_PEGMC = pw["PEG-MC", "CG"], holm_MC_PEGMC = pw["PEG-MC", "MC"])
}
status_of <- function(kw_q, adj_q) factor(case_when(kw_q < 0.05 & adj_q < 0.05 ~ "Robust (both q < 0.05)",
                                                    kw_q < 0.05 | adj_q < 0.05 ~ "One test only", TRUE ~ "Not significant"),
                                          levels = c("Robust (both q < 0.05)", "One test only", "Not significant"))
panel_tests <- map_dfr(names(REG), function(rg) {
  L <- lab_of(rg)
  map_dfr(KEYS, ~ bind_cols(tibble(region = rg, key = .x), test_one(L[[.x]], L$scaffold, L$run))) %>%
    mutate(kw_q = p.adjust(kw_p, "BH"), run_adj_q = p.adjust(run_adj_p, "BH"), status = status_of(kw_q, run_adj_q))
}) %>% left_join(mutate(select(PANEL, key, theme, label), key = as.character(key)), by = "key") %>%
  mutate(key = factor(key, levels = KEYS)) %>% relocate(region, theme, key, label)
write_csv(mutate(panel_tests, across(where(is.numeric), ~ signif(.x, 4))), file.path(OUT, "radiomics_panel_tests.csv"))
ROB <- levels(panel_tests$status)[1]
for (rg in names(REG)) with(filter(panel_tests, region == rg), cat(sprintf(
  "%s: %d/%d robust | CG vs MC %d, CG vs PEG-MC %d, MC vs PEG-MC %d (Holm < 0.05 among robust)\n", rg,
  sum(status == ROB), length(status), sum(holm_CG_MC[status == ROB] < 0.05), sum(holm_CG_PEGMC[status == ROB] < 0.05),
  sum(holm_MC_PEGMC[status == ROB] < 0.05))))

# All extracted features, with the size-dependence flag (exploratory table only — not plotted).
all_feats <- setdiff(sub("^whole_", "", grep("^whole_", names(raw), value = TRUE)), c("region_voxels", "voxels", "volume_mm3"))
all_tests <- map_dfr(c("interior", "whole"), function(rg) {
  L <- filter(raw, !control); vol <- L[[paste0(rg, "_volume_mm3")]]
  centre <- function(v) ave(rank(v, na.last = "keep"), L$scaffold, FUN = function(r) r - mean(r, na.rm = TRUE))
  map_dfr(all_feats, function(f) {
    y <- L[[paste0(rg, "_", f)]]
    if (sum(is.finite(y)) < 12 || sd(y, na.rm = TRUE) == 0) return(NULL)
    ct <- suppressWarnings(cor.test(centre(vol), centre(y), method = "spearman", exact = FALSE))
    bind_cols(tibble(region = rg, feature = f, rho_volume = suppressWarnings(cor(vol, y, method = "spearman", use = "complete.obs")),
                     rho_volume_within_material = unname(ct$estimate), p_volume_within_material = ct$p.value),
              test_one(y, L$scaffold, L$run))
  }) %>% mutate(kw_q = p.adjust(kw_p, "BH"), run_adj_q = p.adjust(run_adj_p, "BH"),
                size_dependent = abs(rho_volume_within_material) > 0.5 & p_volume_within_material < 0.01)
})
write_csv(mutate(all_tests, across(where(is.numeric), ~ signif(.x, 4))), file.path(OUT, "radiomics_all_feature_tests.csv"))

# ── Figure 1: heatmap, rows split by material, panel features in fixed theme order ──
heat <- function(rg, X = NULL) {
  L <- lab_of(rg)
  Z <- if (is.null(X)) sapply(KEYS, function(k) rank_norm(L[[k]])) else apply(X, 2, rank_norm)
  Z[is.na(Z)] <- 0
  rownames(Z) <- L$scan; colnames(Z) <- PANEL$short
  row_ord <- unlist(lapply(levels(L$scaffold), function(l) {
    idx <- which(L$scaffold == l); idx[hclust(dist(Z[idx, ]), "ward.D2")$order] }))
  Z <- Z[row_ord, ]
  ann_c <- data.frame(Theme = PANEL$theme, row.names = colnames(Z))
  ann_r <- data.frame(Scaffold = L$scaffold[row_ord], row.names = rownames(Z))
  pheatmap(Z, color = colorRampPalette(c("#2a78d6", "#f0efec", "#e34948"))(101), breaks = seq(-2.2, 2.2, length.out = 102),
           cluster_cols = FALSE, cluster_rows = FALSE,
           gaps_col = head(cumsum(table(PANEL$theme)), -1), gaps_row = head(cumsum(table(L$scaffold)), -1),
           annotation_col = ann_c, annotation_row = ann_r, annotation_colors = list(Scaffold = PAL, Theme = THEME_COL),
           angle_col = 90, cellwidth = 13.5, cellheight = 7, fontsize = 6.2, fontsize_row = 5.5, fontsize_col = 6.5,
           border_color = NA, fontfamily = FONT, silent = TRUE)
}
for (rg in names(REG)) save_fig(wrap_elements(full = heat(rg)$gtable), paste0("RadFig1_heatmap_", rg), 180, 128)

# ── Figure 2: every panel feature as a boxplot ────────────────────────────────
box_panel <- function(rg, k, tag) {
  info <- filter(PANEL, key == k); tt <- filter(panel_tests, region == rg, key == k)
  df <- lab_of(rg) %>% transmute(scaffold, run, y = .data[[k]]) %>% filter(is.finite(y))
  dc <- ctrl_of(rg) %>% transmute(scaffold, run, y = .data[[k]]) %>% filter(is.finite(y))
  ally <- c(df$y, dc$y)
  use_log <- all(ally > 0) && max(ally) / min(ally) > 30
  pw <- wilcox_test(df, y ~ scaffold, p.adjust.method = "holm")
  yv <- if (use_log) log10(ally) else ally
  top <- max(yv); rng <- diff(range(yv)); if (rng == 0) rng <- 1
  pos <- top + rng * (0.12 + 0.24 * (seq_len(nrow(pw)) - 1))
  pw$y.position <- if (use_log) 10^pos else pos
  pw$label <- paste("p", p_fmt(pw$p.adj))
  lab_fn <- scales::label_number(scale_cut = scales::cut_short_scale())
  p <- ggplot(df, aes(scaffold, y)) +
    geom_boxplot(aes(fill = scaffold), outlier.shape = NA, width = 0.55, linewidth = 0.3, alpha = 0.35, colour = INK2) +
    geom_beeswarm(aes(colour = scaffold, shape = run), size = 1.1, cex = 2.6) +
    geom_point(data = dc, aes(scaffold, y), shape = 4, colour = MUTED, size = 1.3, stroke = 0.5, position = position_nudge(x = 0.38)) +
    scale_fill_manual(values = PAL, guide = "none") + scale_colour_manual(values = PAL, guide = "none") +
    scale_shape_manual(values = SHP, name = "Run") +
    stat_pvalue_manual(pw, label = "label", tip.length = 0.01, size = 2.1, bracket.size = 0.3) +
    labs(x = NULL, y = NULL, tag = tag, title = paste(strwrap(info$label, width = 23), collapse = "\n"),
         subtitle = sprintf("%s\nq %s | run-adj. q %s", info$theme, p_fmt(tt$kw_q), p_fmt(tt$run_adj_q))) +
    theme(plot.title = element_text(size = 7, face = "bold", colour = INK, lineheight = 0.95), plot.title.position = "panel",
          plot.subtitle = element_text(size = 6.1, colour = INK2, lineheight = 1.0), axis.text.x = element_text(size = 7))
  if (use_log) p + scale_y_log10(labels = lab_fn, expand = expansion(mult = c(0.05, 0.10))) else
    p + scale_y_continuous(labels = lab_fn, expand = expansion(mult = c(0.05, 0.10)))
}
for (rg in names(REG)) {
  panels <- map2(KEYS, LETTERS[seq_along(KEYS)], ~ box_panel(rg, .x, .y))
  fig <- wrap_plots(panels, ncol = 4) + guide_area() + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  save_fig(fig, paste0("RadFig2_panel_", rg), 180, 262)
}

# ── Figure 3: effect size per feature (both regions) + PCA of the panel ───────
es <- panel_tests %>% mutate(region = factor(region, levels = c("interior", "whole"),
                                             labels = c("Interior", "Whole scaffold")),
                             short = factor(PANEL$short[match(key, PANEL$key)], levels = rev(PANEL$short)))
f3a <- ggplot(es, aes(epsilon2, short)) +
  geom_line(aes(group = short), colour = LIGHT, linewidth = 0.4) +
  geom_point(aes(shape = region, colour = status), size = 1.9, stroke = 0.7) +
  facet_grid(rows = vars(theme), scales = "free_y", space = "free_y", switch = "y") +
  scale_shape_manual(values = c(Interior = 16, `Whole scaffold` = 1), name = NULL) +
  scale_colour_manual(values = c(`Robust (both q < 0.05)` = INK, `One test only` = MUTED, `Not significant` = LIGHT),
                      labels = c("Robust", "One test", "n.s."), name = NULL) +
  scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0.02, 0.06))) +
  labs(x = expression("Effect size, Kruskal–Wallis " * epsilon^2), y = NULL, tag = "A") +
  theme(strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 1, size = 6.8),
        panel.spacing.y = unit(2, "mm"), legend.position = "bottom", legend.box = "vertical",
        legend.spacing.y = unit(0, "mm"), axis.text.y = element_text(size = 6.8),
        panel.grid.major.x = element_line(colour = "#e1e0d9", linewidth = 0.25))
L <- lab_of("interior")
Zp <- sapply(KEYS, function(k) rank_norm(L[[k]])); Zp[is.na(Zp)] <- 0
pc <- prcomp(Zp, center = TRUE, scale. = FALSE); ve <- 100 * pc$sdev^2 / sum(pc$sdev^2)
sc <- as_tibble(pc$x[, 1:2]) %>% mutate(scaffold = L$scaffold, run = L$run)
r2 <- function(v, g) summary(lm(v ~ g))$r.squared
R2 <- c(pc1_mat = r2(sc$PC1, sc$scaffold), pc1_run = r2(sc$PC1, sc$run), pc2_mat = r2(sc$PC2, sc$scaffold), pc2_run = r2(sc$PC2, sc$run))
f3b <- ggplot(sc, aes(PC1, PC2)) +
  stat_ellipse(aes(colour = scaffold), type = "norm", level = 0.68, linewidth = 0.4, linetype = "22") +
  geom_point(aes(colour = scaffold, shape = run), size = 1.8) +
  scale_colour_manual(values = PAL, name = NULL) + scale_shape_manual(values = SHP, name = "Run") +
  labs(x = sprintf("PC1 (%.0f%%)", ve[1]), y = sprintf("PC2 (%.0f%%)", ve[2]), tag = "B",
       subtitle = sprintf("Interior, 21 features\nPC1: material R² %.2f, run R² %.2f\nPC2: material R² %.2f, run R² %.2f",
                          R2["pc1_mat"], R2["pc1_run"], R2["pc2_mat"], R2["pc2_run"])) +
  theme(legend.position = "bottom", legend.box = "vertical", legend.spacing.y = unit(0, "mm"))
save_fig(wrap_elements(full = f3a) + wrap_elements(full = f3b) + plot_layout(widths = c(1.6, 1)), "RadFig3_summary", 180, 125)

# ── Batch (run) effects: ComBat harmonisation + run-blocked PERMANOVA ─────────
# Every material is present in every run (CG 3/3/3, MC 6/3/5, PEG-MC 3/3/6), so run
# is not confounded with material and can be modelled. Two complementary steps:
#   TEST      PERMANOVA on the UNCORRECTED 21-feature matrix, material adjusted for
#             run (marginal term) with permutations restricted within run. This is the
#             valid inference: testing data that were first corrected with the group
#             protected exaggerates group differences (Nygaard et al., Biostatistics 2016).
#   DISPLAY   ComBat (sva; Johnson et al. 2007) with material as a protected covariate,
#             for the PCA / heatmap. A second ComBat run blind to material is the
#             conservative check: it cannot manufacture a material difference.
suppressPackageStartupMessages({ library(sva); library(vegan) })
quiet <- function(expr) { out <- NULL; invisible(capture.output(out <- suppressMessages(expr))); out }
tf <- function(y) if (all(y > 0, na.rm = TRUE) && max(y, na.rm = TRUE) / min(y, na.rm = TRUE) > 30) log10(y) else y
batch_data <- function(rg) {
  L <- as.data.frame(lab_of(rg)) %>% mutate(run = droplevels(run))
  X <- sapply(KEYS, function(k) tf(L[[k]]))
  for (j in seq_len(ncol(X))) { bad <- !is.finite(X[, j])
    if (any(bad)) X[bad, j] <- ave(X[, j], L$scaffold, FUN = function(v) median(v[is.finite(v)]))[bad] }
  rownames(X) <- L$scan
  list(L = L, raw = X,
       combat = t(quiet(ComBat(dat = t(X), batch = L$run, mod = model.matrix(~ scaffold, data = L), par.prior = TRUE))),
       blind  = t(quiet(ComBat(dat = t(X), batch = L$run, mod = NULL, par.prior = TRUE))))
}
permanova <- function(X, L, nperm = 9999) {
  D <- dist(scale(X)); set.seed(20261117)
  a_mat <- adonis2(D ~ run + scaffold, data = L, by = "margin", permutations = how(nperm = nperm, blocks = L$run))
  set.seed(20261117)
  a_run <- adonis2(D ~ run + scaffold, data = L, by = "margin", permutations = how(nperm = nperm, blocks = L$scaffold))
  set.seed(20261117)
  disp <- permutest(betadisper(D, L$scaffold), permutations = 999)$tab[1, "Pr(>F)"]
  tibble(term = c("run", "material"), R2 = c(a_run["run", "R2"], a_mat["scaffold", "R2"]),
         F = c(a_run["run", "F"], a_mat["scaffold", "F"]), p = c(a_run["run", "Pr(>F)"], a_mat["scaffold", "Pr(>F)"]),
         dispersion_p = disp)
}
pairwise_permanova <- function(X, L) {
  pr <- combn(levels(L$scaffold), 2, simplify = FALSE)
  map_dfr(pr, function(g) {
    keep <- L$scaffold %in% g; Ls <- droplevels(L[keep, ])
    r <- permanova(X[keep, , drop = FALSE], Ls)
    tibble(pair = paste(g, collapse = " vs "), R2 = r$R2[2], F = r$F[2], p = r$p[2])
  }) %>% mutate(p_holm = p.adjust(p, "holm"))
}
pcs <- function(X, L) { pc <- prcomp(scale(X)); ve <- 100 * pc$sdev^2 / sum(pc$sdev^2)
  list(sc = tibble(PC1 = pc$x[, 1], PC2 = pc$x[, 2], scaffold = L$scaffold, run = L$run), ve = ve) }

BATCH <- list(); PERM <- list(); PAIR <- list()
for (rg in names(REG)) {
  B <- batch_data(rg); BATCH[[rg]] <- B
  PERM[[rg]] <- bind_rows(mutate(permanova(B$raw, B$L), data = "uncorrected (run in the model) — the test"),
                          mutate(permanova(B$combat, B$L), data = "after ComBat, material protected — descriptive"),
                          mutate(permanova(B$blind, B$L), data = "after ComBat, blind to material — conservative")) %>%
    mutate(region = rg, .before = 1)
  PAIR[[rg]] <- mutate(pairwise_permanova(B$raw, B$L), region = rg, .before = 1)
  write_csv(bind_cols(select(B$L, scan, scaffold, run), as_tibble(signif(B$combat, 6))),
            file.path(dirname(OUT), paste0("scaffold_radiomics_panel_combat_", rg, ".csv")))
  save_fig(wrap_elements(full = heat(rg, B$combat)$gtable), paste0("RadFig1_heatmap_", rg, "_batchcorrected"), 180, 128)

  pm <- PERM[[rg]]; pw <- PAIR[[rg]]
  r2 <- function(d, t) 100 * pm$R2[grepl(d, pm$data) & pm$term == t]
  before <- pcs(B$raw, B$L); after <- pcs(B$combat, B$L)
  cen <- before$sc %>% group_by(run) %>% summarise(PC1 = mean(PC1), PC2 = mean(PC2))
  base <- list(scale_colour_manual(values = PAL, name = NULL), scale_shape_manual(values = SHP, name = "Run"),
               theme(legend.position = "bottom", legend.box = "horizontal"))
  fa <- ggplot(before$sc, aes(PC1, PC2)) +
    stat_ellipse(aes(group = run), type = "norm", level = 0.68, colour = MUTED, linewidth = 0.4, linetype = "22") +
    geom_point(aes(colour = scaffold, shape = run), size = 1.8) +
    geom_label(data = cen, aes(label = run), size = 2.2, colour = INK2, fill = "white", linewidth = 0, label.padding = unit(0.6, "mm"), family = FONT) +
    labs(x = sprintf("PC1 (%.0f%%)", before$ve[1]), y = sprintf("PC2 (%.0f%%)", before$ve[2]), tag = "A", title = "Before correction",
         subtitle = sprintf("Run explains %.0f%% of feature variance, material %.0f%%\nGrey ellipses: runs", r2("uncorrected", "run"), r2("uncorrected", "material"))) + base
  fb <- ggplot(after$sc, aes(PC1, PC2)) +
    stat_ellipse(aes(colour = scaffold), type = "norm", level = 0.68, linewidth = 0.4, linetype = "22") +
    geom_point(aes(colour = scaffold, shape = run), size = 1.8) +
    labs(x = sprintf("PC1 (%.0f%%)", after$ve[1]), y = sprintf("PC2 (%.0f%%)", after$ve[2]), tag = "B", title = "After ComBat (run removed)",
         subtitle = sprintf("Run explains %.0f%%, material %.0f%%\nColoured ellipses: materials", r2("material protected", "run"), r2("material protected", "material"))) + base
  bars <- pm %>% mutate(stage = factor(case_when(grepl("uncorrected", data) ~ "Before", grepl("protected", data) ~ "ComBat", TRUE ~ "ComBat,\nblind to material"),
                                       levels = c("Before", "ComBat", "ComBat,\nblind to material")),
                        term = factor(ifelse(term == "run", "Run (batch)", "Material"), levels = c("Run (batch)", "Material")))
  fc <- ggplot(bars, aes(stage, 100 * R2, fill = term)) +
    geom_col(position = position_dodge(0.7), width = 0.62, colour = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.0f%%", 100 * R2)), position = position_dodge(0.7), vjust = -0.4, size = 2.2, colour = INK2, family = FONT) +
    scale_fill_manual(values = c(`Run (batch)` = LIGHT, Material = INK2), name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = NULL, y = "Feature variance explained (%)", tag = "C", title = "What drives the 21 features") +
    theme(legend.position = "bottom")
  fd <- ggplot(pw, aes(100 * R2, factor(pair, levels = rev(pair)))) +
    geom_col(width = 0.55, fill = INK2) +
    geom_text(aes(label = paste0("Holm p ", p_fmt(p_holm))), hjust = -0.08, size = 2.3, colour = INK2, family = FONT) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.45))) +
    labs(x = "Variance explained by material, run-adjusted (%)", y = NULL, tag = "D", title = "Pairwise separation",
         subtitle = sprintf("Overall material effect: PERMANOVA p %s", p_fmt(pm$p[grepl("uncorrected", pm$data) & pm$term == "material"])))
  ttl <- theme(plot.title = element_text(size = 8, face = "bold", colour = INK))
  save_fig(((fa + ttl) | (fb + ttl)) / ((fc + ttl) | (fd + ttl)) + plot_layout(heights = c(1.15, 1)),
           paste0("RadFig4_batch_correction_", rg), 180, 165)
  cat(sprintf("%s PERMANOVA (uncorrected, run-adjusted): material R2 %.3f p %s | run R2 %.3f p %s | dispersion p %s\n", rg,
              pm$R2[2], p_fmt(pm$p[2]), pm$R2[1], p_fmt(pm$p[1]), p_fmt(pm$dispersion_p[1])))
  cat(sprintf("   pairwise: %s\n", paste(sprintf("%s R2 %.3f Holm p %s", pw$pair, pw$R2, p_fmt(pw$p_holm)), collapse = "; ")))
}
write_csv(bind_rows(bind_rows(PERM) %>% transmute(region, analysis = data, comparison = term, R2, F, p, p_holm = NA, dispersion_p),
                    bind_rows(PAIR) %>% transmute(region, analysis = "uncorrected (run in the model) — pairwise", comparison = pair, R2, F, p, p_holm, dispersion_p = NA)) %>%
            mutate(across(where(is.numeric), ~ signif(.x, 4))), file.path(OUT, "radiomics_batch_permanova.csv"))

# ── legends ───────────────────────────────────────────────────────────────────
cnt <- function(rg) sum(filter(panel_tests, region == rg)$status == ROB)
robust_names <- function(rg) paste(filter(panel_tests, region == rg, status == ROB) %>% arrange(desc(epsilon2)) %>% pull(label), collapse = "; ")
pt <- filter(panel_tests, region == "interior", status == ROB)
legends <- c(
  "# Radiomic comparison of scaffold materials — figure legends", "",
  sprintf("Data: %d PLL-AuNP-labelled scaffolds from runs R2–R4 (CG n = %d, MC n = %d, PEG-MC n = %d); unlabelled controls (n = %d) are drawn as grey crosses and are not tested. Features: the pre-specified 21-feature panel of the rabbit Defect vs Soaked-OPG vs C-OPG figures, in the same five themes, extracted with the same code (6_extract_features.py) at 0.074 mm voxels; texture on 32 grey levels over a fixed −600 to 1000 HU window. 'Dense' = above 226 HU (mineral particles and gold); scaffolds contain no bone and have no reference ring, so there is no defect : reference ratio. CG scaffolds are smaller than MC and PEG-MC (about 7.5 mm versus 8.1–8.3 mm across), so the three features that scale with region size by definition are expressed per unit volume (marked *): component count per mm³, NGTDM coarseness × voxels, NGTDM busyness per 1,000 voxels. Regions: interior (deeper than 0.25 mm below the seeded surface, i.e. the material without the labelled-cell crust; primary) and whole scaffold. Tests: Kruskal–Wallis across materials with epsilon² effect size and Benjamini–Hochberg FDR over the 21 features; run-adjusted type-II ANOVA of the rank-normalised feature on scaffold + run; 'robust' = q < 0.05 in both; brackets are Holm-adjusted pairwise Wilcoxon p.",
          nrow(L), n_lab["CG"], n_lab["MC"], n_lab["PEG-MC"], nrow(ctrl_of("interior"))), "",
  "**RadFig1. Feature panel by material.** Rank-based inverse-normal scores of the 21 panel features (columns, fixed theme order) for every labelled scan (rows), split into CG, MC and PEG-MC blocks with scans clustered within their own material. One file per region.", "",
  "**RadFig2. Every panel feature.** Boxes: median and interquartile range; points: scans (symbol = run); grey crosses: unlabelled controls of the same material. Under each feature name: its theme, then the FDR q of the Kruskal–Wallis test and, after the bar, of the run-adjusted test. One file per region.", "",
  sprintf("**RadFig3. Summary.** (A) Size of the difference between materials for each feature, in the interior (filled; material without the cell crust) and the whole scaffold (open); shade marks the test outcome (robust = q < 0.05 in both tests). (B) Principal components of the 21 features in the interior with 68%% normal ellipses per material. Robust features: %d of 21 in the interior, %d of 21 in the whole scaffold. Among the interior robust features the pairwise differences are CG vs MC in %d, CG vs PEG-MC in %d and MC vs PEG-MC in %d. Interior robust features, largest effect first: %s.",
          cnt("interior"), cnt("whole"), sum(pt$holm_CG_MC < 0.05), sum(pt$holm_CG_PEGMC < 0.05), sum(pt$holm_MC_PEGMC < 0.05), robust_names("interior")), "")
pmi <- PERM[["interior"]]; pwi <- PAIR[["interior"]]
legends <- c(legends,
  sprintf("**RadFig4. Batch (run) effects and separation of the materials.** Run is the culture and scanning session (R2, R3, R4); every material is present in every run. (A) Principal components of the 21 features before correction; grey ellipses mark runs. (B) After ComBat harmonisation of run with material as a protected covariate; coloured ellipses mark materials. (C) Share of total feature variance explained by run and by material (marginal PERMANOVA R², Euclidean distance on standardised features) before correction, after ComBat, and after a ComBat run blind to material. (D) Pairwise separation of the materials from the run-adjusted PERMANOVA on the uncorrected data, with permutations restricted within run and Holm correction. Inference comes from the uncorrected, run-adjusted model: interior, material R² = %.2f, p %s; run R² = %.2f, p %s; dispersion test p %s. ComBat-corrected data are for display only, because testing data corrected with the group protected overstates group differences. Pairwise (interior): %s. One file per region; RadFig1_heatmap_<region>_batchcorrected shows the corrected features.",
          pmi$R2[2], p_fmt(pmi$p[2]), pmi$R2[1], p_fmt(pmi$p[1]), p_fmt(pmi$dispersion_p[1]),
          paste(sprintf("%s R² %.2f, Holm p %s", pwi$pair, pwi$R2, p_fmt(pwi$p_holm)), collapse = "; ")), "")
writeLines(legends, file.path(OUT, "figure_legends.md"))
cat("Done:", OUT, "\n")
