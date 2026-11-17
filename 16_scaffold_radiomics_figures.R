#!/usr/bin/env Rscript
# Radiomic-feature comparison of Dr. Ren's scaffold materials: CG vs MC vs PEG-MC.
#
# Input : Dr. Ren/results/radiomics/scaffold_radiomics.csv  (15_scaffold_radiomics.py —
#         113 features × 2 regions per scan: whole scaffold, and interior > 0.25 mm deep)
# Output: Dr. Ren/results/radiomics/figures/
#   RadFig1_heatmap_<region>     all features, rank-normalised, scans clustered
#   RadFig2_feature_screen       PCA + per-feature effect size vs FDR + per-family tally (whole)
#   RadFig3_top_features_<region> the most discriminating feature of each family, boxplots
#   RadFig4_whole_vs_interior    do the differences survive removing the cell crust?
#   radiomics_feature_tests.csv  every feature × region: medians, tests, effect sizes
#   figure_legends.md
#
# Statistics (labelled scaffolds, runs R2–R4; duplicate rescan excluded):
#   per feature  Kruskal–Wallis across the three materials, epsilon² effect size,
#                Benjamini–Hochberg FDR across features (q);
#   run-adjusted type-II ANOVA of the rank-normalised feature ~ scaffold + run
#                (R4 scaffolds are intrinsically brighter), BH-adjusted;
#   "robust"     q < 0.05 in BOTH tests;
#   boxplots     pairwise Wilcoxon, Holm-adjusted.
# Stack: ggplot2, ggpubr/rstatix, ggbeeswarm, ggrepel, patchwork, pheatmap, ragg/cairo.
#
#   Rscript 16_scaffold_radiomics_figures.R

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(tibble); library(purrr)
  library(ggpubr); library(rstatix); library(patchwork); library(pheatmap)
  library(ggbeeswarm); library(ggrepel); library(grid)
})
options(warn = 1, dplyr.summarise.inform = FALSE)

args  <- commandArgs(trailingOnly = FALSE)
fpath <- sub("--file=", "", args[grepl("^--file=", args)])
REPO  <- if (length(fpath)) normalizePath(dirname(fpath)) else getwd()
ROOT  <- normalizePath(file.path(REPO, "..", "..", "..", "Dr. Ren"))
IN    <- file.path(ROOT, "results", "radiomics", "scaffold_radiomics.csv")
OUT   <- file.path(ROOT, "results", "radiomics", "figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── palette & theme (same as 14_scaffold_pub_figures.R) ───────────────────────
PAL   <- c(CG = "#2a78d6", MC = "#eb6834", `PEG-MC` = "#1baf7a")
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; LIGHT <- "#c3c2b7"; FAINT <- "#e1e0d9"
SHP   <- c(R2 = 16, R3 = 17, R4 = 15)
FAM_COL <- c(`First-order intensity` = "#eda100", `Threshold / dense fraction` = "#e87ba4",
             `GLCM` = "#008300", `Run / zone / dependence` = "#6250d6",
             `Dense-phase morphometry` = "#e34948")
FONT  <- "Arial"
theme_set(theme_pubr(base_size = 8, base_family = FONT, legend = "right") +
  theme(axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3),
        axis.text = element_text(colour = INK2), axis.title = element_text(colour = INK),
        plot.tag = element_text(face = "bold", size = 10),
        plot.subtitle = element_text(size = 7, colour = INK2),
        strip.background = element_blank(), strip.text = element_text(face = "bold", size = 8),
        legend.key.size = unit(3.5, "mm"), legend.text = element_text(size = 7),
        legend.title = element_text(size = 7.5)))
p_fmt <- function(p) ifelse(is.na(p), "n/a", ifelse(p < 0.001, "< 0.001", ifelse(p < 0.01, sprintf("= %.3f", p), sprintf("= %.2f", p))))
save_fig <- function(p, name, w, h) {
  ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = w, height = h, units = "mm", device = cairo_pdf)
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = w, height = h, units = "mm", dpi = 600,
         device = ragg::agg_png, bg = "white")
  cat("  wrote", name, "\n")
}

# ── feature vocabulary ────────────────────────────────────────────────────────
THRESH <- c("bvtv_fixed", "bone_mean_hu", "otsu_threshold_hu", "bvtv_otsu", "multiotsu_t1_hu",
            "multiotsu_t2_hu", "fraction_low", "fraction_mid", "fraction_high")
family <- function(f) case_when(
  grepl("^glcm_", f) ~ "GLCM", grepl("^glrlm_", f) ~ "GLRLM", grepl("^glszm_", f) ~ "GLSZM",
  grepl("^gldm_", f) ~ "GLDM", grepl("^ngtdm_", f) ~ "NGTDM",
  f %in% THRESH ~ "Threshold / dense fraction", grepl("^bone_", f) ~ "Dense-phase morphometry",
  TRUE ~ "First-order intensity")
FAM_LEVELS <- c("First-order intensity", "Threshold / dense fraction", "GLCM", "GLRLM", "GLSZM",
                "GLDM", "NGTDM", "Dense-phase morphometry")
super_family <- function(fam) ifelse(fam %in% c("GLRLM", "GLSZM", "GLDM", "NGTDM"), "Run / zone / dependence", fam)
SPECIAL <- c(bvtv_fixed = "Dense fraction (> 226 HU)", bone_mean_hu = "Dense-phase mean HU",
             bvtv_otsu = "Fraction above Otsu threshold", otsu_threshold_hu = "Otsu threshold",
             multiotsu_t1_hu = "Multi-Otsu lower threshold", multiotsu_t2_hu = "Multi-Otsu upper threshold",
             mean_hu = "Mean intensity", std_hu = "Intensity SD", median_hu = "Median intensity",
             iqr_hu = "Interquartile range", rms_hu = "Root-mean-square intensity",
             mean_abs_dev_hu = "Mean absolute deviation", entropy_bits = "Histogram entropy",
             p10_hu = "10th percentile", p25_hu = "25th percentile", p75_hu = "75th percentile",
             p90_hu = "90th percentile", min_hu = "Minimum intensity", max_hu = "Maximum intensity",
             range_hu = "Intensity range")
pretty <- function(f) {
  out <- ifelse(f %in% names(SPECIAL), SPECIAL[f], f)
  out <- sub("^glcm_", "GLCM ", out); out <- sub("^glrlm_", "GLRLM ", out); out <- sub("^glszm_", "GLSZM ", out)
  out <- sub("^gldm_", "GLDM ", out); out <- sub("^ngtdm_", "NGTDM ", out); out <- sub("^bone_", "Dense-phase ", out)
  out <- sub("_bits$", "", out); out <- sub("_mm3$", " volume", out); out <- sub("_mm2$", "", out); out <- sub("_mm$", "", out)
  out <- sub("_hu$", "", out)
  unname(gsub("_", " ", out))
}
unit_of <- function(f) case_when(grepl("_hu$", f) ~ " (HU)", grepl("_mm3$", f) ~ " (mm³)", grepl("_mm2$", f) ~ " (mm²)",
                                 grepl("_mm$", f) ~ " (mm)", grepl("_bits$", f) ~ " (bits)", TRUE ~ "")

# ── data ──────────────────────────────────────────────────────────────────────
raw <- read_csv(IN, col_types = cols(scan = col_character(), run = col_character(), batch = col_character(),
                                     scaffold = col_character(), labeled = col_character(), note = col_character(),
                                     .default = col_double()), progress = FALSE) %>%
  filter(!grepl("duplicate", coalesce(note, ""))) %>%
  mutate(control = labeled %in% c("none", "unlabeled"),
         scaffold = factor(scaffold, levels = names(PAL)),
         main = run %in% names(SHP), run = factor(run, levels = c("R1", names(SHP))))
feats_all <- sub("^whole_", "", grep("^whole_", names(raw), value = TRUE))
feats_all <- setdiff(feats_all, c("region_voxels", "voxels", "volume_mm3"))   # region size is not a texture/intensity feature
lab  <- filter(raw, main, !control) %>% mutate(run = droplevels(run))
ctrl <- filter(raw, main, control) %>% mutate(run = droplevels(run))
n_lab <- table(lab$scaffold)
cat(sprintf("labelled R2–R4: %d scans (%s); controls: %d; %d features × 2 regions\n", nrow(lab),
            paste(names(n_lab), n_lab, sep = "=", collapse = ", "), nrow(ctrl), length(feats_all)))

rank_norm <- function(x) { r <- rank(x, na.last = "keep"); qnorm((r - 0.5) / sum(!is.na(x))) }

# ── per-feature tests ─────────────────────────────────────────────────────────
test_region <- function(region) {
  map_dfr(feats_all, function(f) {
    y <- lab[[paste0(region, "_", f)]]
    ok <- is.finite(y)
    if (sum(ok) < 12 || sd(y[ok]) == 0) return(NULL)
    df <- tibble(y = y[ok], scaffold = lab$scaffold[ok], run = lab$run[ok])
    kw <- kruskal.test(y ~ scaffold, data = df)
    df$z <- rank_norm(df$y)
    av <- tryCatch(car::Anova(lm(z ~ scaffold + run, data = df), type = 2), error = function(e) NULL)
    pw <- suppressWarnings(pairwise.wilcox.test(df$y, df$scaffold, p.adjust.method = "holm", exact = FALSE))$p.value
    med <- tapply(df$y, df$scaffold, median)
    tibble(region = region, feature = f, family = family(f), n = nrow(df),
           median_CG = med[["CG"]], median_MC = med[["MC"]], median_PEGMC = med[["PEG-MC"]],
           kw_H = unname(kw$statistic), kw_p = kw$p.value, epsilon2 = unname(kw$statistic) / (nrow(df) - 1),
           run_adj_F = if (is.null(av)) NA else av["scaffold", "F value"],
           run_adj_p = if (is.null(av)) NA else av["scaffold", "Pr(>F)"],
           run_F = if (is.null(av)) NA else av["run", "F value"],
           run_p = if (is.null(av)) NA else av["run", "Pr(>F)"],
           holm_CG_MC = pw["MC", "CG"], holm_CG_PEGMC = pw["PEG-MC", "CG"], holm_MC_PEGMC = pw["PEG-MC", "MC"])
  }) %>% mutate(kw_q = p.adjust(kw_p, "BH"), run_adj_q = p.adjust(run_adj_p, "BH"),
                run_q = p.adjust(run_p, "BH"),
                status = factor(case_when(kw_q < 0.05 & run_adj_q < 0.05 ~ "Robust (both q < 0.05)",
                                          kw_q < 0.05 | run_adj_q < 0.05 ~ "One test only",
                                          TRUE ~ "Not significant"),
                                levels = c("Robust (both q < 0.05)", "One test only", "Not significant")),
                family = factor(family, levels = FAM_LEVELS))
}
tests <- bind_rows(test_region("whole"), test_region("interior"))
write_csv(mutate(tests, across(where(is.numeric), ~ signif(.x, 4))), file.path(OUT, "radiomics_feature_tests.csv"))
tw <- filter(tests, region == "whole"); ti <- filter(tests, region == "interior")
cat(sprintf("whole: %d/%d robust; interior: %d/%d robust; run effect (whole) q<0.05: %d\n",
            sum(tw$status == levels(tw$status)[1]), nrow(tw), sum(ti$status == levels(ti$status)[1]), nrow(ti),
            sum(tw$run_q < 0.05, na.rm = TRUE)))

# ── Figure 1: heatmap of all features ─────────────────────────────────────────
feature_matrix <- function(region, feats) {
  M <- as.matrix(lab[, paste0(region, "_", feats)]); rownames(M) <- lab$scan; colnames(M) <- feats
  Z <- apply(M, 2, rank_norm); Z[is.na(Z)] <- 0; rownames(Z) <- lab$scan
  Z
}
heat <- function(region) {
  tt <- filter(tests, region == !!region)
  feats <- tt$feature
  Z <- feature_matrix(region, feats)
  sf <- factor(super_family(as.character(tt$family)), levels = names(FAM_COL))
  ord <- unlist(lapply(levels(sf), function(l) {          # cluster within family, keep families contiguous
    idx <- which(sf == l); if (length(idx) < 3) return(idx)
    idx[hclust(dist(t(Z[, idx])), "ward.D2")$order] }))
  Z <- Z[, ord]; sf <- sf[ord]
  gaps <- head(cumsum(table(sf)[levels(sf)]), -1)
  ann_c <- data.frame(Family = sf, row.names = colnames(Z))
  ann_r <- data.frame(Scaffold = lab$scaffold, Run = lab$run, row.names = lab$scan)
  pheatmap(Z, color = colorRampPalette(c("#2a78d6", "#f0efec", "#e34948"))(101), breaks = seq(-2.2, 2.2, length.out = 102),
           cluster_cols = FALSE, cluster_rows = TRUE, clustering_method = "ward.D2", gaps_col = gaps,
           annotation_col = ann_c, annotation_row = ann_r,
           annotation_colors = list(Scaffold = PAL, Run = c(R2 = "#9ec5f4", R3 = "#5598e7", R4 = "#1c5cab"), Family = FAM_COL),
           show_colnames = FALSE, fontsize = 6.5, fontsize_row = 5.5, border_color = NA, treeheight_row = 18,
           fontfamily = FONT, silent = TRUE)
}
for (rg in c("whole", "interior")) {
  h <- heat(rg)
  save_fig(wrap_elements(full = h$gtable), paste0("RadFig1_heatmap_", rg), 180, 105)
}

# ── PCA helper ────────────────────────────────────────────────────────────────
pca_plot <- function(region, tag = NULL) {
  tt <- filter(tests, region == !!region)
  Z <- feature_matrix(region, tt$feature)
  pc <- prcomp(Z, center = TRUE, scale. = FALSE)
  ve <- 100 * pc$sdev^2 / sum(pc$sdev^2)
  sc <- as_tibble(pc$x[, 1:2]) %>% mutate(scan = rownames(Z), scaffold = lab$scaffold, run = lab$run)
  # does material or run explain the leading components?
  r2 <- function(v, g) summary(lm(v ~ g))$r.squared
  sub <- sprintf("PC1: material R² %.2f, run R² %.2f  |  PC2: material R² %.2f, run R² %.2f",
                 r2(sc$PC1, sc$scaffold), r2(sc$PC1, sc$run), r2(sc$PC2, sc$scaffold), r2(sc$PC2, sc$run))
  p <- ggplot(sc, aes(PC1, PC2)) +
    stat_ellipse(aes(colour = scaffold), type = "norm", level = 0.68, linewidth = 0.4, linetype = "22") +
    geom_point(aes(colour = scaffold, shape = run), size = 1.8) +
    scale_colour_manual(values = PAL, name = NULL) + scale_shape_manual(values = SHP, name = "Run") +
    labs(x = sprintf("PC1 (%.0f%%)", ve[1]), y = sprintf("PC2 (%.0f%%)", ve[2]), subtitle = sub, tag = tag) +
    theme(legend.position = "bottom", legend.box = "horizontal")
  list(plot = p, ve = ve, r2 = c(pc1_mat = r2(sc$PC1, sc$scaffold), pc1_run = r2(sc$PC1, sc$run),
                                 pc2_mat = r2(sc$PC2, sc$scaffold), pc2_run = r2(sc$PC2, sc$run)))
}

# ── Figure 2: feature screen (whole scaffold) ────────────────────────────────
STAT_COL <- c(`Robust (both q < 0.05)` = INK, `One test only` = MUTED, `Not significant` = LIGHT)
pcw <- pca_plot("whole", "A")
top_lab <- tw %>% filter(status == levels(status)[1]) %>% arrange(desc(epsilon2)) %>% slice_head(n = 7)
f2b <- ggplot(tw, aes(epsilon2, -log10(kw_q))) +
  geom_hline(yintercept = -log10(0.05), linetype = "22", colour = MUTED, linewidth = 0.3) +
  geom_point(aes(colour = status), size = 1.2, alpha = 0.9) +
  geom_text_repel(data = top_lab, aes(label = pretty(feature)), size = 2.1, colour = INK2, family = FONT,
                  segment.size = 0.2, segment.colour = MUTED, min.segment.length = 0, max.overlaps = Inf,
                  direction = "y", hjust = 1, nudge_x = -0.22, xlim = c(NA, 0.36), box.padding = 0.3, force = 3) +
  annotate("text", x = min(tw$epsilon2), y = -log10(0.05), label = "q = 0.05", hjust = 0, vjust = -0.5, size = 2.1, colour = MUTED) +
  scale_colour_manual(values = STAT_COL, name = NULL) +
  labs(x = expression("Effect size, Kruskal–Wallis " * epsilon^2), y = expression(-log[10] * " FDR q"), tag = "B",
       subtitle = "Each point is one feature; shade as in C") +
  theme(legend.position = "none")
tally <- tw %>% count(family, status) %>% complete(family, status, fill = list(n = 0))
f2c <- ggplot(tally, aes(n, forcats::fct_rev(family), fill = forcats::fct_rev(status))) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.4) +
  scale_fill_manual(values = c(`Not significant` = FAINT, `One test only` = MUTED, `Robust (both q < 0.05)` = INK),
                    name = NULL, guide = guide_legend(reverse = TRUE)) +
  labs(x = "Number of features", y = NULL, tag = "C") + theme(legend.position = "bottom")
fig2 <- wrap_elements(full = (pcw$plot | f2b)) / f2c + plot_layout(heights = c(1, 0.62))
save_fig(fig2, "RadFig2_feature_screen", 180, 150)

# ── Figure 3: top feature per family ──────────────────────────────────────────
top_panel <- function(region, f, tag) {
  col <- paste0(region, "_", f)
  tt <- filter(tests, region == !!region, feature == f)
  df <- lab %>% transmute(scaffold, run, y = .data[[col]]) %>% filter(is.finite(y))
  dc <- ctrl %>% transmute(scaffold, run, y = .data[[col]]) %>% filter(is.finite(y))
  ally <- c(df$y, dc$y)
  use_log <- all(ally > 0) && max(ally) / min(ally) > 30
  pw <- wilcox_test(df, y ~ scaffold, p.adjust.method = "holm")
  yv <- if (use_log) log10(ally) else ally
  top <- max(yv); rng <- diff(range(yv)); if (rng == 0) rng <- 1
  pos <- top + rng * (0.08 + 0.13 * (seq_len(nrow(pw)) - 1))
  pw$y.position <- if (use_log) 10^pos else pos
  pw$label <- paste("p", p_fmt(pw$p.adj))
  p <- ggplot(df, aes(scaffold, y)) +
    geom_boxplot(aes(fill = scaffold), outlier.shape = NA, width = 0.55, linewidth = 0.3, alpha = 0.35, colour = INK2) +
    geom_beeswarm(aes(colour = scaffold, shape = run), size = 1.2, cex = 2.6) +
    geom_point(data = dc, aes(scaffold, y), shape = 4, colour = MUTED, size = 1.4, stroke = 0.5,
               position = position_nudge(x = 0.38)) +
    scale_fill_manual(values = PAL, guide = "none") + scale_colour_manual(values = PAL, guide = "none") +
    scale_shape_manual(values = SHP, name = "Run") +
    stat_pvalue_manual(pw, label = "label", tip.length = 0.01, size = 2.2, bracket.size = 0.3) +
    labs(x = NULL, y = paste0(pretty(f), unit_of(f)), tag = tag,
         title = sub("Threshold / dense fraction", "Threshold-based", sub("Dense-phase morphometry", "Dense-phase shape", as.character(tt$family))),
         subtitle = sprintf("q %s | run-adj. q %s", p_fmt(tt$kw_q), p_fmt(tt$run_adj_q))) +
    theme(plot.title = element_text(size = 7.5, face = "bold", colour = INK), plot.title.position = "panel",
          plot.subtitle = element_text(size = 6.3, colour = INK2), axis.title.y = element_text(size = 7))
  lab_fn <- scales::label_number(scale_cut = scales::cut_short_scale())
  if (use_log) p + scale_y_log10(labels = lab_fn, expand = expansion(mult = c(0.05, 0.06))) else
    p + scale_y_continuous(labels = lab_fn, expand = expansion(mult = c(0.05, 0.06)))
}
top_fig <- function(region) {
  pick <- filter(tests, region == !!region) %>% group_by(family) %>% arrange(run_adj_p, desc(epsilon2)) %>% slice_head(n = 1) %>% ungroup() %>% arrange(family)
  panels <- map2(pick$feature, LETTERS[seq_len(nrow(pick))], ~ top_panel(region, .x, .y))
  wrap_plots(panels, ncol = 4) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
}
for (rg in c("whole", "interior")) save_fig(top_fig(rg), paste0("RadFig3_top_features_", rg), 180, 135)

# ── Figure 4: whole vs interior ───────────────────────────────────────────────
both <- inner_join(select(tw, feature, family, e_whole = epsilon2, s_whole = status),
                   select(ti, feature, e_int = epsilon2, s_int = status), by = "feature") %>%
  mutate(cls = factor(case_when(s_whole == levels(s_whole)[1] & s_int == levels(s_int)[1] ~ "Robust in both regions",
                                s_whole == levels(s_whole)[1] ~ "Robust in whole scaffold only",
                                s_int == levels(s_int)[1] ~ "Robust in interior only", TRUE ~ "Neither"),
                      levels = c("Robust in both regions", "Robust in whole scaffold only", "Robust in interior only", "Neither")))
rho <- cor(both$e_whole, both$e_int, method = "spearman")
f4a <- ggplot(both, aes(e_whole, e_int)) +
  geom_abline(slope = 1, intercept = 0, colour = MUTED, linewidth = 0.3, linetype = "22") +
  geom_point(aes(colour = cls, shape = cls), size = 1.4) +
  scale_colour_manual(values = c(`Robust in both regions` = INK, `Robust in whole scaffold only` = MUTED,
                                 `Robust in interior only` = MUTED, Neither = LIGHT), name = NULL) +
  scale_shape_manual(values = c(`Robust in both regions` = 16, `Robust in whole scaffold only` = 17,
                                `Robust in interior only` = 15, Neither = 16), name = NULL) +
  labs(x = expression("Effect size " * epsilon^2 * ", whole scaffold"), y = expression("Effect size " * epsilon^2 * ", interior (> 0.25 mm deep)"),
       subtitle = sprintf("Spearman ρ = %.2f across %d features", rho, nrow(both)), tag = "A") +
  coord_equal() + theme(legend.position = "bottom", legend.direction = "vertical")
pci <- pca_plot("interior", "B")
fig4 <- (f4a | pci$plot) + plot_layout(widths = c(1, 1.1))
save_fig(fig4, "RadFig4_whole_vs_interior", 180, 100)

# ── legends ───────────────────────────────────────────────────────────────────
rob_w <- sum(tw$status == levels(tw$status)[1]); rob_i <- sum(ti$status == levels(ti$status)[1])
topn <- function(t, k = 5) paste(pretty(head(arrange(filter(t, status == levels(t$status)[1]), run_adj_p)$feature, k)), collapse = "; ")
legends <- c(
  "# Radiomic comparison of scaffold materials — figure legends", "",
  sprintf("Data: %d PLL-AuNP-labelled scaffolds from runs R2–R4 (CG n = %d, MC n = %d, PEG-MC n = %d); unlabelled controls (n = %d, one per material and run) are drawn as grey crosses and are not tested. %d radiomic features per region (first-order intensity, threshold-based, GLCM, GLRLM, GLSZM, GLDM, NGTDM, dense-phase morphometry) extracted with the feature code of the rabbit pipeline (6_extract_features.py) on 0.074 mm voxels; texture on 32 grey levels over a fixed −600 to 1000 HU window; the 226 HU threshold selects the dense phase (mineral particles and gold). Regions: whole scaffold volume, and interior (deeper than 0.25 mm below the seeded surface, i.e. without the labelled-cell crust). Per feature: Kruskal–Wallis across materials with epsilon² effect size and Benjamini–Hochberg FDR; and a run-adjusted type-II ANOVA of the rank-normalised feature on scaffold + run. 'Robust' = q < 0.05 in both.",
          nrow(lab), n_lab["CG"], n_lab["MC"], n_lab["PEG-MC"], nrow(ctrl), length(feats_all)), "",
  "**RadFig1. All radiomic features.** Rank-based inverse-normal scores per feature (columns, grouped by family and clustered within family) for every labelled scan (rows, Ward clustering); row strips give material and run. One file per region.", "",
  sprintf("**RadFig2. Feature screen, whole scaffold.** (A) Principal components of the rank-normalised feature matrix; 68%% normal ellipses per material; the subtitle gives the share of each component explained by material and by run. (B) Effect size against FDR for each feature; labelled points are the seven robust features with the largest effect. (C) Number of features per family by test outcome. %d of %d features are robust. Strongest: %s.",
          rob_w, nrow(tw), topn(tw)), "",
  "**RadFig3. The most discriminating feature of each family.** Boxes: median and interquartile range; points: scans (symbol = run); grey crosses: unlabelled controls of the same material. Brackets: Holm-adjusted pairwise Wilcoxon p. Subtitles give the feature's FDR q for the Kruskal–Wallis test and, after the bar, for the run-adjusted test. One file per region.", "",
  sprintf("**RadFig4. Are the differences intrinsic to the material?** (A) Effect size of each feature in the whole scaffold against the interior region that excludes the cell crust (dashed line = identity). (B) Principal components for the interior region. %d of %d interior features are robust. Strongest: %s.",
          rob_i, nrow(ti), topn(ti)), "")
writeLines(legends, file.path(OUT, "figure_legends.md"))
cat("Done:", OUT, "\n")
