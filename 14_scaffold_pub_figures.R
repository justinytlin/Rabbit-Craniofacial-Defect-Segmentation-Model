#!/usr/bin/env Rscript
# Publication figures for Dr. Ren's nanogold scaffold study — group comparisons
# of the metrics extracted by the web app (scaffold tab of radiomics_database).
#
# Stack: ggplot2 + ggpubr/rstatix (tests and brackets), ggbeeswarm, patchwork,
# pheatmap (multivariate view), ragg/cairo (600 dpi PNG + vector PDF, Arial).
#
# Figures (Dr. Ren/results/publication/):
#   Fig1_label_detection      labelled vs unlabelled, per scaffold, + threshold robustness
#   Fig2_group_comparison     labelled R2–R4: six metrics across CG / MC / PEG-MC
#   Fig3_depth                depth profiles (median, IQR) + D90
#   Fig4_multivariate         z-score heatmap of all metrics + PCA
#   Fig5_run_dose             gold amount by run / dose, per scaffold
#   statistics.csv            every test behind the figures
#   figure_legends.md         legends with n, tests and corrections
#
# Statistics: two-group → Wilcoxon rank-sum (Mann–Whitney); three groups →
# Kruskal–Wallis, then pairwise Wilcoxon with Holm correction. R2–R4 only
# (8 mm scaffolds); R1 (6 mm pilot) shown where stated but never tested; the
# duplicate rescan 6070 is excluded.
#
#   Rscript 14_scaffold_pub_figures.R [--out DIR]

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(tibble)
  library(ggpubr); library(rstatix); library(patchwork); library(pheatmap)
  library(ggbeeswarm); library(ggrepel); library(grid)
})
options(warn = 1, dplyr.summarise.inform = FALSE)

args  <- commandArgs(trailingOnly = FALSE)
fpath <- sub("--file=", "", args[grepl("^--file=", args)])
REPO  <- if (length(fpath)) normalizePath(dirname(fpath)) else getwd()
DB    <- file.path(REPO, "radiomics_database", "scaffold_database.csv")
ROOT  <- normalizePath(file.path(REPO, "..", "..", "..", "Dr. Ren"))
targs <- commandArgs(trailingOnly = TRUE)
OUT   <- if ("--out" %in% targs) targs[which(targs == "--out") + 1] else file.path(ROOT, "results", "publication")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── palette & theme ───────────────────────────────────────────────────────────
PAL   <- c(CG = "#2a78d6", MC = "#eb6834", `PEG-MC` = "#1baf7a")
CTRL  <- "#c3c2b7"; INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"
SHP   <- c(R1 = 1, R2 = 16, R3 = 17, R4 = 15)
FONT  <- "Arial"
theme_pub <- theme_pubr(base_size = 8, base_family = FONT, legend = "right") +
  theme(axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3),
        axis.text = element_text(colour = INK2), axis.title = element_text(colour = INK),
        plot.tag = element_text(face = "bold", size = 10),
        plot.subtitle = element_text(size = 7.5, colour = INK2),
        strip.background = element_blank(), strip.text = element_text(face = "bold", size = 8),
        legend.key.size = unit(3.5, "mm"), legend.text = element_text(size = 7),
        legend.title = element_text(size = 7.5), panel.spacing = unit(3, "mm"))
theme_set(theme_pub)

logy <- function(...) scale_y_log10(
  breaks = function(l) 10^seq(floor(log10(l[1])), ceiling(log10(l[2]))),
  labels = scales::trans_format("log10", scales::math_format(10^.x)), ...)
p_fmt <- function(p) ifelse(p < 0.001, "p < 0.001", ifelse(p < 0.01, sprintf("p = %.3f", p), sprintf("p = %.2f", p)))
STATS <- list()
log_stat <- function(fig, panel, tbl) STATS[[length(STATS) + 1]] <<- mutate(tbl, figure = fig, panel = panel, .before = 1)

# ── data ──────────────────────────────────────────────────────────────────────
num_cols <- c("cells", "replicate", "diameter_mm", "k", "threshold_raw", "baseline_mode", "sigma",
              "scaffold_volume_mm3", "analysed_thickness_mm", "top_cut_ratio", "gold_vox", "gold_frac_pct",
              "gold_volume_mm3", "gold_intensity_sum", "D50_mm", "D50w_mm", "D90_mm", "mean_depth_mm",
              "frac_surface", "frac_0p25_1mm", "frac_deeper_1mm", "deep_density_pct", "mean_rnorm",
              "frac_core", "n_clusters", "largest_cluster_frac", "median_cluster_vox", "frac_gold_bottom15",
              "k4_gold_frac_pct", "k6_gold_frac_pct", "k6_frac_deeper_1mm", "k6_deep_density_pct",
              "T3000_gold_frac_pct", "T3000_frac_deeper_1mm")
d <- read_csv(DB, col_types = cols(.default = col_character()), progress = FALSE) %>%
  mutate(across(all_of(num_cols), as.numeric)) %>%
  filter(!grepl("duplicate", coalesce(note, ""))) %>%
  mutate(control  = labeled %in% c("none", "unlabeled"),
         group    = factor(ifelse(control, "Control", "Labelled"), levels = c("Control", "Labelled")),
         scaffold = factor(scaffold, levels = names(PAL)),
         run      = factor(run, levels = names(SHP)),
         main     = run %in% c("R2", "R3", "R4"))
main <- filter(d, main)
lab  <- filter(main, !control)
n_lab <- table(lab$scaffold)
cat(sprintf("%d scans; R2–R4: %d labelled (%s), %d controls\n", nrow(d), nrow(lab),
            paste(names(n_lab), n_lab, sep = "=", collapse = ", "), sum(main$control)))

# ── helpers ───────────────────────────────────────────────────────────────────
# Three-group comparison: Kruskal–Wallis + pairwise Wilcoxon (Holm), bracket positions.
three_group <- function(df, var, log = FALSE) {
  df <- mutate(df, .y = .data[[var]])
  kw <- kruskal_test(df, .y ~ scaffold)
  pw <- wilcox_test(df, .y ~ scaffold, p.adjust.method = "holm")
  yv <- if (log) log10(df$.y) else df$.y
  top <- max(yv, na.rm = TRUE); rng <- diff(range(yv, na.rm = TRUE)); if (!is.finite(rng) || rng == 0) rng <- 1
  pos <- top + rng * (0.08 + 0.13 * (seq_len(nrow(pw)) - 1))
  pw$y.position <- if (log) 10^pos else pos
  pw$label <- p_fmt(pw$p.adj)
  list(kw = kw, pw = pw)
}

panel_three <- function(df, var, ylab, tag, log = FALSE, floor = NULL, scale = 1, show_shape = FALSE) {
  df <- mutate(df, .y = .data[[var]] * scale)
  if (!is.null(floor)) df$.y <- pmax(df$.y, floor)
  t <- three_group(df, ".y", log)
  log_stat("Fig2/3", paste(tag, var), bind_rows(
    tibble(test = "Kruskal-Wallis", group1 = "all", group2 = "", n = t$kw$n, statistic = t$kw$statistic, p = t$kw$p, p.adj = NA),
    transmute(t$pw, test = "Wilcoxon pairwise (Holm)", group1, group2, n = n1 + n2, statistic, p, p.adj)))
  xl <- function(x) x
  p <- ggplot(df, aes(scaffold, .y)) +
    geom_boxplot(aes(fill = scaffold), outlier.shape = NA, width = 0.55, linewidth = 0.3,
                 alpha = 0.35, colour = INK2) +
    geom_beeswarm(aes(colour = scaffold, shape = run), size = 1.3, cex = 2.6) +
    scale_fill_manual(values = PAL, guide = "none") +
    scale_colour_manual(values = PAL, guide = "none") +
    scale_shape_manual(values = SHP, name = "Run", drop = TRUE) +
    scale_x_discrete(labels = xl) +
    stat_pvalue_manual(t$pw, label = "label", tip.length = 0.01, size = 2.3, bracket.size = 0.3,
                       colour = INK2) +
    labs(x = NULL, y = ylab, tag = tag, subtitle = paste("Kruskal–Wallis", p_fmt(t$kw$p)))
  p <- if (log) p + logy(expand = expansion(mult = c(0.05, 0.06))) else if (scale == 100)
    p + scale_y_continuous(breaks = seq(0, 100, 25), expand = expansion(mult = c(0.05, 0.06))) else
    p + scale_y_continuous(expand = expansion(mult = c(0.05, 0.06)))
  if (!show_shape) p <- p + guides(shape = "none")
  p
}

save_fig <- function(p, name, w, h) {
  ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = w, height = h, units = "mm", device = cairo_pdf)
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = w, height = h, units = "mm", dpi = 600,
         device = ragg::agg_png, bg = "white")
  cat("  wrote", name, "\n")
}

# ── Figure 1: label detection vs controls ─────────────────────────────────────
panel_lab_ctrl <- function(df, var, ylab, floor, tag, ceiling = NULL, bracket_y = NULL) {
  df <- df %>% mutate(.y = pmax(.data[[var]], floor),
                      fillkey = ifelse(control, "control", as.character(scaffold)))
  tests <- df %>% group_by(scaffold) %>% wilcox_test(.y ~ group) %>%
    adjust_pvalue(method = "holm") %>% mutate(label = p_fmt(p.adj))
  tests$y.position <- if (is.null(bracket_y)) 10^(max(log10(df$.y)) + 0.35) else bracket_y
  log_stat("Fig1", paste(tag, var), transmute(tests, test = "Wilcoxon labelled vs control (Holm over scaffolds)",
                                               group1 = paste(scaffold, group1), group2 = paste(scaffold, group2),
                                               n = n1 + n2, statistic, p, p.adj))
  p <- ggplot(df, aes(group, .y)) +
    geom_boxplot(aes(fill = fillkey), outlier.shape = NA, width = 0.6, linewidth = 0.3, alpha = 0.45, colour = INK2) +
    geom_beeswarm(aes(shape = run), colour = INK2, size = 1.1, cex = 2.8) +
    facet_wrap(~scaffold, nrow = 1) +
    scale_fill_manual(values = c(control = CTRL, PAL), guide = "none") +
    scale_shape_manual(values = SHP, name = "Run") +
    scale_x_discrete(labels = c(Control = "\u2212", Labelled = "+")) +
    logy(expand = expansion(mult = c(0.05, 0.1))) +
    stat_pvalue_manual(tests, label = "label", tip.length = 0.01, size = 2.3, bracket.size = 0.3, colour = INK2) +
    labs(x = "PLL-AuNP label", y = ylab, tag = tag) +
    theme(axis.text.x = element_text(size = 9))
  if (!is.null(ceiling))
    p <- p + geom_hline(yintercept = ceiling, linetype = "22", colour = MUTED, linewidth = 0.3)
  p
}
f1a <- panel_lab_ctrl(main, "gold_frac_pct", "Gold-like voxels (% of scaffold volume)", 2e-4, "A",
                      ceiling = 0.02, bracket_y = 12) +
  geom_text(data = tibble(scaffold = factor("CG", levels = names(PAL)), group = factor("Control", levels = c("Control", "Labelled"))),
            aes(x = 0.5, y = 0.02), label = "control ceiling", hjust = 0, vjust = -0.4, size = 2.2, colour = MUTED,
            inherit.aes = FALSE)
f1b <- panel_lab_ctrl(main, "deep_density_pct", "Gold > 1 mm deep (% of deep scaffold)", 1e-5, "B", bracket_y = 0.5)

thr <- main %>%
  transmute(scan, scaffold, group, run, `k = 4` = k4_gold_frac_pct, `k = 5` = gold_frac_pct,
            `k = 6` = k6_gold_frac_pct, `fixed 3000` = T3000_gold_frac_pct) %>%
  pivot_longer(c(`k = 4`, `k = 5`, `k = 6`, `fixed 3000`), names_to = "threshold", values_to = ".y") %>%
  mutate(threshold = factor(threshold, levels = c("k = 4", "k = 5", "k = 6", "fixed 3000")), .y = pmax(.y, 2e-4))
thr_t <- thr %>% group_by(threshold) %>% wilcox_test(.y ~ group) %>% adjust_pvalue(method = "holm") %>%
  mutate(label = p_fmt(p.adj), y.position = 30)
log_stat("Fig1", "C threshold robustness", transmute(thr_t, test = "Wilcoxon labelled vs control (Holm over thresholds)",
                                                    group1 = paste(threshold, group1), group2 = paste(threshold, group2),
                                                    n = n1 + n2, statistic, p, p.adj))
f1c <- ggplot(thr, aes(threshold, .y, fill = group)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, linewidth = 0.3, alpha = 0.5, colour = INK2,
               position = position_dodge(0.75)) +
  geom_point(aes(group = group), colour = INK2, size = 0.8, alpha = 0.7,
             position = position_jitterdodge(jitter.width = 0.18, dodge.width = 0.75, seed = 1)) +
  scale_fill_manual(values = c(Control = CTRL, Labelled = "#c98500"), name = NULL) +
  logy(expand = expansion(mult = c(0.05, 0.1))) +
  stat_pvalue_manual(thr_t, label = "label", x = "threshold", tip.length = 0, size = 2.3, bracket.size = 0,
                     colour = INK2) +
  geom_hline(yintercept = 0.02, linetype = "22", colour = MUTED, linewidth = 0.3) +
  labs(x = "Gold threshold (per-slice baseline + k·σ, or fixed raw value)",
       y = "Gold-like voxels (% of scaffold volume)", tag = "C") +
  theme(legend.position = c(0.9, 0.18))
fig1 <- (f1a + f1b) / (f1c) + plot_layout(heights = c(1, 0.85), guides = "collect") &
  theme(legend.position = "bottom", legend.box = "horizontal")
save_fig(fig1, "Fig1_label_detection", 180, 135)

# ── Figure 2: labelled scaffolds compared ─────────────────────────────────────
f2 <- list(
  panel_three(lab, "gold_frac_pct", "Gold-like voxels (% of scaffold)", "A", log = TRUE),
  panel_three(lab, "gold_volume_mm3", expression("Gold volume (mm"^3*")"), "B", log = TRUE),
  panel_three(lab, "D50_mm", "Median gold depth, D50 (mm)", "C"),
  panel_three(lab, "frac_surface", "Gold within 0.25 mm of surface (%)", "D", scale = 100),
  panel_three(lab, "deep_density_pct", "Gold > 1 mm deep (% of deep scaffold)", "E", log = TRUE, floor = 1e-5),
  panel_three(lab, "mean_rnorm", "Mean radial position (0 centre – 1 edge)", "F"),
  panel_three(lab, "frac_core", "Gold in central half-radius (%)", "G", scale = 100),
  panel_three(lab, "largest_cluster_frac", "Largest cluster (% of gold)", "H", scale = 100, show_shape = TRUE)
)
fig2 <- wrap_plots(f2, ncol = 4) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
save_fig(fig2, "Fig2_group_comparison", 180, 125)

# ── Figure 3: depth ───────────────────────────────────────────────────────────
prof <- lapply(seq_len(nrow(lab)), function(i) {
  f <- file.path(ROOT, "outputs", paste0(lab$scan[i], "_scaffold"), paste0(lab$scan[i], "_scaffold_depth_profile.csv"))
  if (!file.exists(f)) return(NULL)
  p <- read_csv(f, col_types = cols(.default = col_double()), progress = FALSE)
  if (!nrow(p)) return(NULL)
  mutate(p, scan = lab$scan[i], scaffold = lab$scaffold[i], depth = depth_lo_mm + 0.05, cum = cumsum(frac_of_gold))
}) %>% bind_rows()
prof_s <- prof %>% group_by(scaffold, depth) %>%
  summarise(med = median(local_gold_pct), q1 = quantile(local_gold_pct, .25), q3 = quantile(local_gold_pct, .75),
            cmed = median(cum), cq1 = quantile(cum, .25), cq3 = quantile(cum, .75), n = n()) %>% ungroup()
FL <- 1e-4
f3a <- ggplot(prof_s, aes(depth, pmax(med, FL), colour = scaffold, fill = scaffold)) +
  geom_ribbon(aes(ymin = pmax(q1, FL), ymax = pmax(q3, FL)), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.6) +
  geom_vline(xintercept = c(0.25, 1), linetype = "22", colour = MUTED, linewidth = 0.3) +
  scale_colour_manual(values = PAL, name = NULL) + scale_fill_manual(values = PAL, name = NULL) +
  scale_y_log10(breaks = c(1e-4, 1e-3, 1e-2, 1e-1, 1, 10), labels = c("≤0.0001", "0.001", "0.01", "0.1", "1", "10")) +
  scale_x_continuous(breaks = seq(0, 3, 0.5)) +
  labs(x = "Depth below seeded surface (mm)", y = "Local gold density (% of scaffold voxels)", tag = "A")
f3b <- ggplot(prof_s, aes(depth, 100 * cmed, colour = scaffold, fill = scaffold)) +
  geom_ribbon(aes(ymin = 100 * cq1, ymax = 100 * cq3), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.6) +
  geom_vline(xintercept = c(0.25, 1), linetype = "22", colour = MUTED, linewidth = 0.3) +
  scale_colour_manual(values = PAL, name = NULL) + scale_fill_manual(values = PAL, name = NULL) +
  scale_x_continuous(breaks = seq(0, 3, 0.5)) + scale_y_continuous(limits = c(0, 100)) +
  labs(x = "Depth below seeded surface (mm)", y = "Cumulative share of gold (%)", tag = "B")
f3c <- panel_three(lab, "D90_mm", "90th-percentile gold depth, D90 (mm)", "C", show_shape = TRUE)
fig3 <- (f3a | f3b | f3c) + plot_layout(widths = c(1.15, 1.15, 0.9), guides = "collect") &
  theme(legend.position = "bottom", legend.box = "horizontal")
save_fig(fig3, "Fig3_depth", 180, 72)

# ── Figure 4: multivariate ────────────────────────────────────────────────────
mvars <- c(`Gold fraction (log)` = "gold_frac_pct", `Gold volume (log)` = "gold_volume_mm3",
           `Gold intensity sum (log)` = "gold_intensity_sum", `D50 depth` = "D50_mm", `D90 depth` = "D90_mm",
           `Mean depth` = "mean_depth_mm", `Within 0.25 mm` = "frac_surface", `0.25–1 mm` = "frac_0p25_1mm",
           `Deeper than 1 mm` = "frac_deeper_1mm", `Deep density (log)` = "deep_density_pct",
           `Radial position` = "mean_rnorm", `Core share` = "frac_core", `Clusters (log)` = "n_clusters",
           `Largest cluster share` = "largest_cluster_frac", `Median cluster size (log)` = "median_cluster_vox")
M <- lab %>% select(scan, all_of(unname(mvars))) %>% column_to_rownames("scan") %>% as.matrix()
colnames(M) <- names(mvars)
logc <- grepl("\\(log\\)", colnames(M))
M[, logc] <- log10(pmax(M[, logc], 1e-5))
M <- M[, apply(M, 2, function(x) sd(x, na.rm = TRUE) > 0), drop = FALSE]
M[is.na(M)] <- 0
Z <- scale(M)
ann <- data.frame(Scaffold = lab$scaffold, Run = lab$run, row.names = lab$scan)
ann$Run <- droplevels(ann$Run)
ann_col <- list(Scaffold = PAL, Run = c(R2 = "#9ec5f4", R3 = "#5598e7", R4 = "#1c5cab"))
hm <- pheatmap(Z, color = colorRampPalette(c("#2a78d6", "#f0efec", "#e34948"))(101),
               breaks = seq(-2.5, 2.5, length.out = 102), annotation_row = ann, annotation_colors = ann_col,
               clustering_method = "ward.D2", border_color = NA, fontsize = 6.5, fontsize_row = 5.5,
               fontsize_col = 6.5, treeheight_row = 18, treeheight_col = 14, angle_col = 45,
               legend = TRUE, annotation_legend = TRUE, silent = TRUE, fontfamily = FONT,
               main = "")
pc <- prcomp(Z, center = FALSE, scale. = FALSE)
ve <- 100 * pc$sdev^2 / sum(pc$sdev^2)
sc <- as_tibble(pc$x[, 1:2]) %>% mutate(scan = rownames(Z), scaffold = lab$scaffold, run = lab$run)
ld <- as_tibble(pc$rotation[, 1:2], rownames = "var") %>% mutate(norm = sqrt(PC1^2 + PC2^2)) %>%
  arrange(desc(norm)) %>% slice_head(n = 7)
sf <- 0.8 * min(diff(range(sc$PC1)) / diff(range(ld$PC1)), diff(range(sc$PC2)) / diff(range(ld$PC2)))
ld <- mutate(ld, x = PC1 * sf, y = PC2 * sf)
f4b <- ggplot(sc, aes(PC1, PC2)) +
  stat_ellipse(aes(colour = scaffold), type = "norm", level = 0.68, linewidth = 0.4, linetype = "22") +
  geom_point(aes(colour = scaffold, shape = run), size = 1.7) +
  geom_text_repel(data = filter(sc, PC1 < -5), aes(label = scan), size = 2.1, colour = INK2, family = FONT,
                  nudge_y = -0.8, segment.size = 0.2) +
  geom_segment(data = ld, aes(x = 0, y = 0, xend = x, yend = y), colour = MUTED, linewidth = 0.3,
               arrow = arrow(length = unit(1.2, "mm"))) +
  geom_text_repel(data = ld, aes(x, y, label = var), size = 2.1, colour = INK2, family = FONT,
                  segment.size = 0.2, min.segment.length = 0.2) +
  scale_colour_manual(values = PAL, name = NULL) + scale_shape_manual(values = SHP, name = "Run", drop = TRUE) +
  labs(x = sprintf("PC1 (%.0f%% of variance)", ve[1]), y = sprintf("PC2 (%.0f%% of variance)", ve[2])) +
  coord_equal() + theme(legend.position = "bottom", legend.box = "vertical")
fig4 <- wrap_elements(full = hm$gtable) + f4b + plot_layout(widths = c(1.45, 1)) +
  plot_annotation(tag_levels = "A")
save_fig(fig4, "Fig4_multivariate", 180, 110)
log_stat("Fig4", "PCA", tibble(test = "PCA variance explained", group1 = "PC1", group2 = "PC2",
                               n = nrow(Z), statistic = ve[1], p = ve[2], p.adj = NA))

# ── Figure 5: run / dose ──────────────────────────────────────────────────────
d5 <- mutate(d, .y = pmax(gold_frac_pct, 2e-4),
             run_lab = factor(run, levels = names(SHP),
                              labels = c("R1\n6 mm pilot", "R2\n500k", "R3\n500k", "R4\n1M")))
aov2 <- car::Anova(lm(log10(gold_frac_pct) ~ scaffold + run, data = lab), type = 2)
log_stat("Fig5", "ANOVA", tibble(test = "Type-II ANOVA log10 gold fraction (R2–R4 labelled)",
                                 group1 = rownames(aov2), group2 = "", n = nrow(lab),
                                 statistic = aov2$`F value`, p = aov2$`Pr(>F)`, p.adj = NA))
fig5 <- ggplot(filter(d5, !control), aes(run_lab, .y)) +
  geom_boxplot(aes(fill = scaffold), outlier.shape = NA, width = 0.55, linewidth = 0.3, alpha = 0.35, colour = INK2) +
  geom_beeswarm(aes(colour = scaffold, shape = run), size = 1.3, cex = 2.6) +
  geom_point(data = filter(d5, control), aes(run_lab, .y), shape = 4, colour = MUTED, size = 1.3, stroke = 0.5) +
  geom_hline(yintercept = 0.02, linetype = "22", colour = MUTED, linewidth = 0.3) +
  facet_wrap(~scaffold, nrow = 1) +
  scale_fill_manual(values = PAL, guide = "none") + scale_colour_manual(values = PAL, guide = "none") +
  scale_shape_manual(values = SHP, name = "Run", drop = TRUE, guide = "none") +
  logy(expand = expansion(mult = c(0.05, 0.1))) +
  labs(x = NULL, y = "Gold-like voxels (% of scaffold volume)",
       subtitle = sprintf("Type-II ANOVA on log gold fraction (R2–R4): run %s, scaffold %s.   Open symbols = R1 pilot; \u00d7 = controls",
                          p_fmt(aov2["run", "Pr(>F)"]), p_fmt(aov2["scaffold", "Pr(>F)"])))
save_fig(fig5, "Fig5_run_dose", 180, 70)

# ── statistics + legends ──────────────────────────────────────────────────────
stats <- bind_rows(STATS) %>% mutate(across(c(statistic, p, p.adj), ~ signif(.x, 4)))
write_csv(stats, file.path(OUT, "statistics.csv"))

n_ctrl <- table(filter(main, control)$scaffold)
legends <- c(
  "# Figure legends", "",
  sprintf("All panels: scans from runs R2–R4 (8 mm scaffolds; %d PLL-AuNP-labelled: CG n = %d, MC n = %d, PEG-MC n = %d; %d unlabelled or no-cell controls, one per scaffold and run). The duplicate rescan 6070 is excluded. Gold = voxels above the per-slice scaffold baseline + 5σ inside the scaffold envelope (threshold calibrated on the 20 controls of all runs so that 95%% of controls hold ≤ 0.02%% gold-like voxels). Boxes: median and interquartile range, whiskers to 1.5 × IQR; every scan is plotted (symbol = run). Two-group tests are Wilcoxon rank-sum; three-group tests are Kruskal–Wallis followed by pairwise Wilcoxon with Holm correction; p values on brackets are Holm-adjusted. Analysis: 12_scaffold_gold.py (web app) on analysis/microct_pipeline.py; figures: 14_scaffold_pub_figures.R (ggplot2, ggpubr, rstatix, pheatmap).",
          nrow(lab), n_lab["CG"], n_lab["MC"], n_lab["PEG-MC"], sum(main$control)), "",
  "**Figure 1. The gold label is detected and the detection is robust to the threshold.** (A) Gold-like voxels as a percentage of scaffold volume, unlabelled controls versus labelled scaffolds, per scaffold material; dashed line = 0.02% control ceiling. (B) Gold density more than 1 mm below the seeded surface (gold voxels > 1 mm deep as % of scaffold voxels > 1 mm deep); labelled scaffolds do not differ from controls. Zero values are drawn at the axis floor. (C) Labelled-versus-control separation at the primary threshold (k = 5) and at k = 4, k = 6 and a fixed 3000-raw threshold, all scaffolds pooled.", "",
  sprintf("**Figure 2. Labelled scaffolds compared across materials.** (A) Gold fraction, (B) gold volume, (C) median gold depth D50, (D) share of gold within 0.25 mm of the seeded surface, (E) deep gold density below 1 mm, (F) mean normalised radial position of gold (0 = scaffold centre, 1 = edge), (G) share of gold within the central half-radius, (H) share of gold in the largest connected cluster. Kruskal–Wallis p above each panel; brackets give Holm-adjusted pairwise Wilcoxon p. CG n = %d, MC n = %d, PEG-MC n = %d labelled scaffolds.", n_lab["CG"], n_lab["MC"], n_lab["PEG-MC"]), "",
  "**Figure 3. Depth distribution of the label.** (A) Local gold density per 0.1 mm depth bin (median line, interquartile ribbon across scans); (B) cumulative share of gold with depth; dashed lines at 0.25 mm and 1 mm. (C) 90th-percentile gold depth per scan.", "",
  sprintf("**Figure 4. Multivariate view of all extracted metrics.** (A) Heatmap of z-scored metrics (log10 for skewed quantities) for every labelled R2–R4 scan, rows and columns clustered (Ward.D2, Euclidean); row annotation gives scaffold and run. (B) Principal-component scores (PC1 %.0f%%, PC2 %.0f%% of variance) with 68%% normal ellipses per scaffold and the seven strongest loadings; the two labelled outliers (6088, 6092) are the scans flagged in the report for gold at the scaffold floor.", ve[1], ve[2]), "",
  "**Figure 5. Gold amount by run and dose.** Labelled scaffolds by run within each material (R1 = 6 mm pilot with 4e5/8e5 cells, open symbols, descriptive only; R2 and R3 = 500k cells; R4 = 1M cells); × = unlabelled or no-cell controls; dashed line = control ceiling. Type-II ANOVA on log10 gold fraction (R2–R4) tests run and scaffold; dose is confounded with run because R4 was a separate culture and scanning session.", ""
)
writeLines(legends, file.path(OUT, "figure_legends.md"))
cat("Done:", OUT, "\n")
