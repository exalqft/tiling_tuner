require(hadron)
require(dplyr)
require(ggplot2)

archs <- c("dual_epyc2_7742_gcc",
           "nvidia_a100", "amd_mi250")

bws <- data.frame(bw=c(204.8, 2*204.8,
                       1555, 1600),
                  ylim=c(500, 500,
                         1600, 1650),
                  architecture=c("dual_epyc2_7742_gcc", "dual_epyc2_7742_gcc",
                                 "nvidia_a100", "amd_mi250"),
                  nt=c(64, 128,
                       32, 7))

tikzfiles <- hadron::tikz.init("su3_stream_mdrange", width=12, height=6)

for(i in 1:length(archs)) {
  arch <- archs[i]
  bwdat <- dplyr::filter(bws, architecture == arch)
  dat <- read.table(sprintf("%s/results.dat", arch), header=TRUE) %>% dplyr::filter(grepl("su3xsu3", kernel))
  ncol <- NULL
  maxdat <- dplyr::group_by(dat, nt, tiling) %>%
            dplyr::filter(n == max(n)) %>%
            dplyr::filter(bw == max(bw)) %>%
            dplyr::ungroup()

  p <- ggplot2::ggplot(dat , aes(x = 3*n^4*4*9*16*10^(-6), y = bw, colour = tiling, shape = tiling)) +
       ggplot2::geom_line() +
       ggplot2::geom_point() +
       ggplot2::geom_hline(data = bwdat, aes(yintercept = bw), colour = "blue") +
       ggplot2::geom_hline(data = maxdat, aes(yintercept = bw), colour = "red") +
       ggplot2::geom_point(data = bwdat, shape = NA, fill = NA, colour = NA, x = 1,
                           aes(y = bw)) +
       ggplot2::geom_label(data = bwdat,
                           aes(label = sprintf(" mem %0.0f GB/s", bw),
                               y = bw, x = 0.5),
                           colour = "blue",
                           inherit.aes = FALSE,
                           size = 2.5) +
       ggplot2::geom_label(data = maxdat, 
                           aes(label = sprintf("%0.0f GB/s", bw),
                               y = bw, x = 0.5),
                           colour = "red",
                           inherit.aes=FALSE, 
                           size = 2.5) +
       ggplot2::ggtitle(hadron::escapeLatexSpecials(sprintf("%s, OMP_PROC_BIND=close, OMP_PLACES=cores", arch))) +
       ggplot2::facet_wrap(sprintf("$n_{\\textrm{th}}=%03d$",nt) ~ sprintf("kernel = %s", kernel), ncol = ncol) +
       ggplot2::scale_x_continuous(trans = "log10") +
       ggplot2::coord_cartesian(ylim = c(0, bwdat$ylim[1])) +
       ggplot2::labs(x = "gauge field size [MB]",
                     y = "BW [GB/s]") +
       ggplot2::theme_bw() +
       ggplot2::theme(plot.title = element_text(color="blue", size = 10, face="bold")) 
  plot(p)
}

hadron::tikz.finalize(tikzfiles, crop=FALSE)