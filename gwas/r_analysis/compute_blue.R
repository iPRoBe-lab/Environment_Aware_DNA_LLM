#!/usr/bin/env Rscript
# Stage-1 trait values: environment-adjusted genotype BLUEs via a fixed-effects model (base R lm).
#   trait ~ genotype + (env:rep:block)        [all fixed]
#   genotype  -> coefficient is the BLUE; env:rep:block absorbs environment, replicate and block.
# Fast at this scale (sparse-free dense lm, 8 fits in parallel, single-thread BLAS each).
# Output: blue_pheno/{trait}_{cat}.csv (Taxa,<trait>) + _blue_summary.tsv.
suppressMessages({library(data.table); library(parallel)})
setDTthreads(1)
PROJ <- Sys.getenv("PLANT_PROJECT", ".")   # project root: set $PLANT_PROJECT or run from it
OUT  <- file.path(PROJ,"local_experiments","blue_pheno"); dir.create(OUT, showWarnings=FALSE, recursive=TRUE)
TRAITS <- c("yield","anthesis","silking","ASI")
CATS   <- c("NO_STRESS","HEAT_STRESSED","DROUGHT_STRESSED","HEAT_DROUGHT_STRESSED")

ph  <- fread(file.path(PROJ,"project_input_data","PHENO.csv"))
idx <- fread(file.path(PROJ,"project_input_data","Heat_Drought_Stress_Indices.csv"))[, .(year_location, CATEGORY)]
ph  <- merge(ph, idx, by.x="year_loc", by.y="year_location", all.x=TRUE)
stopifnot(!any(is.na(ph$CATEGORY)))

blue_one <- function(tr, ct){
  d <- ph[CATEGORY==ct, .(genotype, year_loc, rep, block, y=get(tr))][!is.na(y)]
  d[, genotype := factor(genotype)]
  d[, erb := factor(paste(year_loc, rep, block, sep="_"))]   # env:rep:block (encodes env & rep)
  m  <- lm(y ~ genotype + erb, data=d)
  co <- coef(m); g <- co[grepl("^genotype", names(co))]
  blue <- data.table(Taxa=c(levels(d$genotype)[1], sub("^genotype","",names(g))), v=c(0, unname(g)))
  n_na <- sum(is.na(blue$v)); blue <- blue[!is.na(v)]
  blue[, v := v - mean(v)]
  naive <- d[, .(naive=mean(y)), by=.(Taxa=genotype)]; naive[, naive := naive - mean(naive)]
  r <- suppressWarnings(cor(blue$v, merge(blue, naive, by="Taxa")$naive))
  out <- blue[, .(Taxa, V=v)]; setnames(out, "V", tr)
  fwrite(out, file.path(OUT, sprintf("%s_%s.csv", tr, ct)))
  data.table(trait=tr, stress=ct, n_geno=nrow(out), n_aliased=n_na, corr_BLUE_vs_naive=round(r,3))
}
jobs <- expand.grid(trait=TRAITS, stress=CATS, stringsAsFactors=FALSE)
t0 <- Sys.time()
res <- mclapply(seq_len(nrow(jobs)), function(i) blue_one(jobs$trait[i], jobs$stress[i]),
                mc.cores=as.integer(Sys.getenv("BLUE_CORES","8")), mc.preschedule=FALSE)
S <- rbindlist(res); fwrite(S, file.path(OUT,"_blue_summary.tsv"), sep="\t")
cat("total seconds:", round(as.numeric(Sys.time()-t0, units="secs"),1), "\n"); print(S)
