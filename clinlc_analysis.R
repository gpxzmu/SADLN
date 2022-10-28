library(parallel)
clin<-read.csv("./analysis/BRCA/clin.csv",header=T)
clust<-read.csv("./analysis/BRCA/clust.csv",header=T)
  
rownames(clin)=clin$X
clin<-clin[,-1]
rownames(clust)=clust$X
clust=as.matrix(clust)
rownames(clust)=rownames(clin)
clust<-clust[,-1]
clinical_metadata = list(gender='DISCRETE', age_at_initial_pathologic_diagnosis='NUMERIC', pathologic_M='DISCRETE', 
                         pathologic_N='DISCRETE', pathologic_T='DISCRETE', pathologic_stage='DISCRETE')


clin <- clin[which(names(clin) %in% names(clinical_metadata))] 
##clin <- clin[which(row.names(clin) %in% rownames(clust)),]
##stopifnot(nrow(clin) == nrow(clust))
##indices <- match(row.names(clust), row.names(clin))
##clin <- clin[indices,]
clin$cluster <- clust

get.empirical.clinical <- function(clustering, clinical.values, is.chisq) {
  set.seed(42)
  if (is.chisq) {
    clustering.with.clinical = cbind(clustering, clinical.values)
    tbl = table(as.data.frame(clustering.with.clinical))
    test.res = chisq.test(tbl)
  } else {
    test.res = kruskal.test(as.numeric(clinical.values), clustering)
  }
  orig.pvalue = test.res$p.value
  num.iter = 1000
  total.num.iters = 0
  total.num.extreme = 0
  should.continue = T
  
  while (should.continue) {
    perm.pvalues = as.numeric(mclapply(1:num.iter, function(i) {
      cur.clustering = sample(clustering)
      names(cur.clustering) = names(clustering)
      
      if (is.chisq) {
        clustering.with.clinical = cbind(cur.clustering, clinical.values)
        tbl = table(as.data.frame(clustering.with.clinical))
        test.res = chisq.test(tbl)
      } else {
        test.res = kruskal.test(as.numeric(clinical.values), cur.clustering)
      }
      cur.pvalue = test.res$p.value
      return(cur.pvalue)
    }))
    total.num.iters = total.num.iters + num.iter
    total.num.extreme = total.num.extreme + sum(perm.pvalues <= orig.pvalue)
    
    binom.ret = binom.test(total.num.extreme, total.num.iters)
    cur.pvalue = binom.ret$estimate
    cur.conf.int = binom.ret$conf.int
    
    sig.threshold = 0.05
    is.threshold.in.conf = cur.conf.int[1] < sig.threshold & cur.conf.int[2] < sig.threshold
    if (!is.threshold.in.conf | total.num.iters > 1e5) {
      should.continue = F
    }
  }
  
  return(cur.pvalue)
}


pvalues = c()
params_tested = c()

for (clinical_param in names(clinical_metadata)) {
  
  print(clinical_param)
  
  if (!(clinical_param %in% colnames(clin))) {
    next
  }
  
  is_discrete_param <- clinical_metadata[clinical_param] == 'DISCRETE' 
  is_numeric_param <- clinical_metadata[clinical_param] == 'NUMERIC'
  stopifnot(is_discrete_param | is_numeric_param)
  
  df = clin[,c(clinical_param, "cluster")]
  
  if (clinical_param == "pahtologic_M") {
    df$pathologic_M = gsub("a", "", df$pathologic_M)
    df$pathologic_M = gsub("b", "", df$pathologic_M)
    df$pathologic_M = gsub("c", "", df$pathologic_M)
    df$pathologic_M = gsub(" \\(i\\+\\)", "", df$pathologic_M)
    df[which(!df$pathologic_M %in% c("MX", "M0", "M1")),1] = NA
  } else if (clinical_param == "pahtologic_T") {
    df[which(df$pathologic_T == "[Discrepancy]"),1] = NA
    df$pathologic_T = gsub("a", "", df$pathologic_T)
    df$pathologic_T = gsub("b", "", df$pathologic_T)
    df$pathologic_T = gsub("c", "", df$pathologic_T)
    df$pathologic_T = gsub("d", "", df$pathologic_T)
    df[which(!df$pathologic_T %in% c("TX", "T0", "T1", "T2", "T3", "T4", "Tis")),1] = NA
  } else if (clinical_param == "pathologic_N") {
    df$pathologic_N = gsub("a", "", df$pathologic_N)
    df$pathologic_N = gsub("b", "", df$pathologic_N)
    df$pathologic_N = gsub("c", "", df$pathologic_N)
    df$pathologic_N = gsub("mi", "", df$pathologic_N)
    df$pathologic_N = gsub(" \\(i\\+\\)", "", df$pathologic_N)
    df$pathologic_N = gsub(" \\(i\\-\\)", "", df$pathologic_N)
    df$pathologic_N = gsub(" \\(mol\\+\\)", "", df$pathologic_N)
    df[which(!df$pathologic_N %in% c("NX", "N1", "N2", "N3", "N0")),1] = NA
  } else if (clinical_param == "pathologic_stage") {
    df[which(df$pathologic_stage == "[Discrepancy]"),1] = NA
    df$pathologic_stage = gsub("A", "", df$pathologic_stage)
    df$pathologic_stage = gsub("B", "", df$pathologic_stage)
    df$pathologic_stage = gsub("C", "", df$pathologic_stage)
    df[which(!df$pathologic_stage %in% c("Stage X", "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV")),1] = NA
  }
  
  
  clinical_values <- df[,clinical_param]
  
  if (is_numeric_param) {
    numeric_entries = !is.na(as.numeric(clinical_values)) # Boolean list > TRUE if not NA and FALSE if is NA
    # if too many missing data (more than half), skip the param
    if (2 * sum(numeric_entries) < length(clinical_values)) {
      next
    }
  } else {
    not_na_entries = !is.na(clinical_values) 
    should_skip = F
    if (2 * sum(not_na_entries) < length(clinical_values)) {
      should_skip = T
    } else if (length(table(clinical_values[not_na_entries])) == 1) {
      should_skip = T
    }
    if (should_skip) {
      next
    }
  }
  
  params_tested = c(params_tested, clinical_param)
  
  clustering_list <- df$cluster
  names(clustering_list) <- row.names(df)
  
  if (is_discrete_param) {
    pvalue = get.empirical.clinical(clustering_list[!is.na(clinical_values)], clinical_values[!is.na(clinical_values)], T)
  } else if (is_numeric_param) {
    pvalue = get.empirical.clinical(clustering_list[numeric_entries], as.numeric(clinical_values[numeric_entries]), F)
  }
  
  pvalues = c(pvalues, pvalue)
  
}

names(pvalues) = params_tested

sum(pvalues < 0.05)

#pvalues
names(pvalues) = params_tested
sum(pvalues < 0.01)
pvalues