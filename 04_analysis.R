seed <- 20260907L
sign_flips <- 10000L
resamples <- 10000L
outcomes <- c("PHQ9", "RSES")
metrics <- c("cosine", "euclidean", "angular", "l1")

args <- commandArgs(trailingOnly = TRUE)
input <- args[1]
output <- args[2]
model_catalogue <- args[3]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
read_matrix <- function(name) {
    read.csv(file.path(input, paste0(name, ".csv")), row.names = "sample_id", check.names = FALSE,
             colClasses = c(sample_id = "character"))
}
save_results <- function(x, name) write.csv(x, file.path(output, paste0(name, ".csv")), row.names = FALSE)
eligible <- read_matrix("participants")
lengths <- eligible$shorter_speech_length
eligible$shorter_speech_length <- NULL
primary <- lengths >= 10
participants <- eligible[primary, ]
linguistic_variables <- setdiff(names(participants), c(outcomes, "age", "male"))
adjustments <- list(
    age_sex_adjusted = c("age", "male"),
    linguistically_adjusted = c("age", "male", linguistic_variables)
)

associations <- function(SED, people, controls) {
    models <- names(SED)
    grid <- expand.grid(model_key = models, outcome = outcomes, stringsAsFactors = FALSE)
    predictors <- sprintf("embedding_%02d", seq_along(models))
    names(SED) <- predictors
    grid$predictor <- rep(predictors, length(outcomes))
    frame <- cbind(SED, people[c(outcomes, controls)])
    continuous <- setdiff(names(frame), "male")
    frame[continuous] <- scale(frame[continuous])
    fits <- lapply(seq_len(nrow(grid)), function(i) {
        formula <- reformulate(c(grid$predictor[i], controls), grid$outcome[i])
        do.call(lm, list(formula = formula, data = frame))
    })
    names(fits) <- sprintf("spec_%03d", seq_along(fits))
    signs <- pima::pima(fits, tested_coeffs = as.list(grid$predictor),
                        n_flips = sign_flips, method = "none", score_type = "standardized", seed = seed)
    joint <- pima::combine_tests(signs, comb_funct = "mean", by = "response", tail = 0)$summary_table
    coefficients <- do.call(rbind, lapply(seq_along(fits), function(i) {
        fit <- fits[[i]]
        predictor <- grid$predictor[i]
        ci <- confint(fit, predictor)
        c(standardized_beta = unname(coef(fit)[predictor]), ci_lower = ci[1], ci_upper = ci[2],
          p = summary(fit)$coefficients[predictor, 4])
    }))
    estimates <- cbind(grid[c("model_key", "outcome")], coefficients, pima_p = signs$summary_table$p)
    panel_summary <- do.call(rbind, lapply(outcomes, function(outcome) {
        block <- estimates[estimates$outcome == outcome, ]
        beta <- block$standardized_beta
        q <- quantile(beta, c(.25, .5, .75), names = FALSE)
        magnitude <- quantile(abs(beta), c(.01, .99), names = FALSE)
        data.frame(outcome, n = nrow(SED), model_count = length(models),
                   positive_count = sum(beta > 0), negative_count = sum(beta < 0),
                   zero_count = sum(beta == 0),
                   median_beta = q[2], beta_q25 = q[1], beta_q75 = q[3], iqr_width = q[3] - q[1],
                   voe_ratio = magnitude[2] / magnitude[1],
                   beta_min = min(beta), beta_max = max(beta), significant_count = sum(block$p < .05),
                   joint_p = joint$p[match(outcome, joint$model)])
    }))
    panel_summary$adjusted_p <- p.adjust(panel_summary$joint_p, "BH")
    list(estimates = estimates, summary = panel_summary)
}

bootstrap_coefficients <- function(SED, people, controls) {
    X <- as.matrix(SED)
    Y <- as.matrix(people[outcomes])
    nuisance <- cbind(1, as.matrix(people[controls]))
    coefficient <- function(data, i) {
        x <- X[i, , drop = FALSE]
        y <- Y[i, , drop = FALSE]
        residual <- qr.resid(qr(nuisance[i, , drop = FALSE]), cbind(x, y))
        rx <- residual[, seq_len(ncol(x)), drop = FALSE]
        ry <- residual[, ncol(x) + seq_along(outcomes), drop = FALSE]
        slopes <- crossprod(rx, ry) / colSums(rx^2)
        spread <- function(a) sqrt(colSums(sweep(a, 2, colMeans(a))^2))
        as.vector(sweep(slopes * spread(x), 2, spread(y), "/"))
    }
    RNGkind("Mersenne-Twister", "Inversion", "Rejection")
    set.seed(seed)
    draws <- boot::boot(people, coefficient, R = resamples)
    colnames(draws$t) <- as.vector(outer(names(SED), outcomes, function(m, o) paste(o, m, sep = "__")))
    draws
}

panel_intervals <- function(draws, models, statistics) {
    intervals <- list()
    for (o in seq_along(outcomes)) {
        columns <- (o - 1) * models + seq_len(models)
        for (name in names(statistics)) {
            statistic <- statistics[[name]]
            observed <- statistic(draws$t0[columns])
            values <- apply(draws$t[, columns, drop = FALSE], 1, statistic)
            b <- list(t0 = observed, t = matrix(values), R = resamples)
            ci <- boot::boot.ci(b, type = "perc", index = 1)$percent
            intervals[[length(intervals) + 1]] <- data.frame(
                outcome = outcomes[o], statistic = name,
                estimate = observed, ci_lower = ci[4], ci_upper = ci[5]
            )
        }
    }
    do.call(rbind, intervals)
}

estimates <- summaries <- intervals <- list()
for (metric in metrics) {
    all_sed <- read_matrix(metric)
    stopifnot(identical(rownames(all_sed), rownames(eligible)))
    SED <- all_sed[primary, , drop = FALSE]
    for (analysis in names(adjustments)) {
        controls <- adjustments[[analysis]]
        result <- associations(SED, participants, controls)
        label <- paste(metric, analysis)
        estimates[[label]] <- data.frame(metric, analysis, result$estimates)
        summaries[[label]] <- data.frame(metric, analysis, result$summary)
        if (metric == "cosine") {
            draws <- bootstrap_coefficients(SED, participants, controls)
            saveRDS(draws$t, file.path(output, paste0("bootstrap_", analysis, ".rds")))
            statistics <- list(median = median)
            if (ncol(SED) > 1) statistics$iqr <- IQR
            ci <- panel_intervals(draws, ncol(SED), statistics)
            intervals[[analysis]] <- data.frame(analysis, ci)
            if (analysis == "age_sex_adjusted") {
                primary_draws <- draws
                primary_result <- result
            }
        }
    }
}
save_results(do.call(rbind, estimates), "model_results")
save_results(do.call(rbind, summaries), "summaries")
save_results(do.call(rbind, intervals), "bootstrap_intervals")

cosine <- read_matrix("cosine")

decompose <- function(data, i) {
    people <- data[i, , drop = FALSE]
    models <- names(cosine)
    nuisance <- qr(cbind(1, people$age, people$male))
    standardized <- scale(people[c(models, linguistic_variables, outcomes)])
    standardized[, attr(standardized, "scaled:scale") == 0] <- 0
    residual <- qr.resid(nuisance, standardized)
    D <- residual[, models, drop = FALSE]
    X <- residual[, linguistic_variables, drop = FALSE]
    Y <- residual[, outcomes, drop = FALSE]
    base <- crossprod(D, Y) / colSums(D^2)
    projection <- qr(X)
    weights <- qr.coef(projection, cbind(D, Y))
    adjusted <- qr.resid(projection, D)
    full <- crossprod(adjusted, qr.resid(projection, Y)) / colSums(adjusted^2)
    gamma <- sweep(crossprod(X, D), 2, colSums(D^2), "/")
    blocks <- lapply(outcomes, function(outcome) {
        contribution <- gamma * (weights[, outcome] - sweep(weights[, models, drop = FALSE], 2, full[, outcome], "*"))
        if (projection$rank < ncol(X)) contribution[,] <- NA_real_
        rbind(age_sex_adjusted_beta = base[, outcome], linguistically_adjusted_beta = full[, outcome],
              coefficient_change = base[, outcome] - full[, outcome], contribution)
    })
    do.call(cbind, blocks)
}

data <- cbind(cosine[primary, , drop = FALSE], participants)
components <- decompose(data, seq_len(nrow(data)))
model_keys <- names(cosine)
gelbach <- expand.grid(component = rownames(components), model_key = model_keys, outcome = outcomes,
                       stringsAsFactors = FALSE)
gelbach$estimate <- as.vector(components)
shares <- sweep(components, 2, components["coefficient_change", ], "/") * 100
shares[!rownames(shares) %in% linguistic_variables, ] <- NA_real_
gelbach$percentage <- as.vector(shares)
save_results(gelbach, "gelbach_model_results")

component_medians <- function(x) {
    vapply(seq_along(outcomes), function(o) {
        columns <- (o - 1L) * length(model_keys) + seq_along(model_keys)
        apply(x[, columns, drop = FALSE], 1, median)
    }, numeric(nrow(x)))
}
indices <- boot::boot.array(primary_draws, indices = TRUE)
replicated <- apply(indices, 1, function(i) as.vector(component_medians(decompose(data, i))))
limits <- t(apply(replicated, 1, quantile, c(.025, .975), type = 6, na.rm = TRUE))
quartiles <- function(x) setNames(quantile(x, c(.5, .25, .75), na.rm = TRUE), c("median", "q25", "q75"))
gelbach_summary <- expand.grid(component = rownames(components), outcome = outcomes, stringsAsFactors = FALSE)
rows <- lapply(seq_len(nrow(gelbach_summary)), function(i) {
    key <- gelbach_summary[i, ]
    block <- gelbach[gelbach$outcome == key$outcome & gelbach$component == key$component, ]
    share <- quartiles(block$percentage)
    data.frame(key, as.list(quartiles(block$estimate)), ci_lower = limits[i, 1], ci_upper = limits[i, 2],
               valid_resamples = sum(is.finite(replicated[i, ])),
               as.list(setNames(share, paste0("percentage_", names(share)))))
})
save_results(do.call(rbind, rows), "gelbach_summaries")

catalogue <- jsonlite::read_json(model_catalogue)
characteristics <- do.call(rbind, lapply(names(catalogue), function(key) {
    fields <- catalogue[[key]]$characteristics
    fields <- lapply(fields, function(value) if (is.null(value)) NA else value)
    data.frame(model_key = key, fields)
}))
release_date <- as.Date(paste0(characteristics$public_release_month, "-01"), format = "%Y-%m-%d")
characteristics$public_release_year <- as.numeric(format(release_date, "%Y")) +
    (as.numeric(format(release_date, "%m")) - 1) / 12
characteristics$log10_parameters_millions <- log10(characteristics$parameter_count_millions)
characteristics <- characteristics[match(names(cosine), characteristics$model_key), ]

bootstrap_estimate <- function(values) {
    draws <- values[-1]
    ci <- quantile(draws, c(.025, .975), type = 6, names = FALSE)
    p <- min(1, 2 * (1 + min(sum(draws <= 0), sum(draws >= 0))) / (1 + length(draws)))
    data.frame(estimate = values[1], ci_lower = ci[1], ci_upper = ci[2], p)
}

group_comparison <- function(beta, group, reference) {
    levels <- sort(setdiff(unique(na.omit(group)), "Unknown"))
    if (!length(levels)) return(NULL)
    if (!reference %in% levels) reference <- levels[1]
    means <- lapply(levels, function(g) rowMeans(beta[, group %in% g, drop = FALSE]))
    names(means) <- levels
    sizes <- vapply(levels, function(g) sum(group %in% g), integer(1))
    rows <- lapply(levels, function(g) data.frame(group = g, model_count = sizes[[g]],
                                                  bootstrap_estimate(means[[g]])))
    contrasts <- lapply(setdiff(levels, reference), function(g) {
        data.frame(term = paste(g, "minus", reference), model_count = sizes[[g]] + sizes[[reference]],
                   bootstrap_estimate(means[[g]] - means[[reference]]))
    })
    if (length(contrasts) > 1) {
        differences <- do.call(cbind, means[setdiff(levels, reference)]) - means[[reference]]
        draws <- differences[-1, , drop = FALSE]
        precision <- solve(cov(draws))
        observed <- drop(crossprod(differences[1, ], precision %*% differences[1, ]))
        centered <- sweep(draws, 2, colMeans(draws))
        null <- rowSums((centered %*% precision) * centered)
        contrasts[[length(contrasts) + 1]] <- data.frame(term = "joint", model_count = sum(sizes),
            estimate = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_,
            p = (1 + sum(null >= observed)) / (1 + length(null)))
    }
    list(means = do.call(rbind, rows), tests = do.call(rbind, contrasts))
}

fit_curves <- function(beta, predictor) {
    keep <- is.finite(predictor)
    x <- predictor[keep]
    center <- mean(x)
    z <- x - center
    terms <- c("intercept", "linear", "quadratic")
    rows <- list()
    for (degree in 1:2) {
        if (length(unique(x)) <= degree) next
        design <- outer(z, 0:degree, `^`)
        fits <- qr.coef(qr(design), t(beta[, keep, drop = FALSE]))
        for (i in seq_len(nrow(fits))) {
            rows[[length(rows) + 1]] <- data.frame(regression = terms[degree + 1], term = terms[i],
                model_count = length(x), center, bootstrap_estimate(fits[i, ]))
        }
    }
    do.call(rbind, rows)
}

references <- c(model_origin = "BERT-based", embedding_structure = "Bidirectional",
                korean_specialization = "No")
tests <- means <- curves <- list()
for (outcome in outcomes) {
    columns <- paste(outcome, characteristics$model_key, sep = "__")
    beta <- rbind(primary_draws$t0, primary_draws$t)[, columns, drop = FALSE]
    for (axis in names(references)) {
        groups <- group_comparison(beta, characteristics[[axis]], references[[axis]])
        label <- paste(outcome, axis)
        if (!is.null(groups$means)) means[[label]] <- data.frame(outcome, axis, groups$means)
        if (!is.null(groups$tests)) tests[[label]] <- data.frame(outcome, axis, groups$tests)
    }
    for (axis in c("log10_parameters_millions", "public_release_year", "STS17")) {
        fit <- fit_curves(beta, characteristics[[axis]])
        if (is.null(fit)) next
        label <- paste(outcome, axis)
        curves[[label]] <- data.frame(outcome, axis, fit)
        tests[[label]] <- data.frame(outcome, axis,
            fit[fit$term == fit$regression, c("term", "model_count", "estimate", "ci_lower", "ci_upper", "p")])
    }
}
if (length(tests)) {
    tests <- do.call(rbind, tests)
    origin_has_joint <- tests$outcome %in% tests$outcome[tests$axis == "model_origin" & tests$term == "joint"]
    family <- !(origin_has_joint & tests$axis == "model_origin" & tests$term != "joint")
    tests$adjusted_p <- NA_real_
    tests$adjusted_p[family] <- p.adjust(tests$p[family], "BH")
    save_results(tests, "characteristic_tests")
}
if (length(means)) save_results(do.call(rbind, means), "characteristic_means")
if (length(curves)) save_results(do.call(rbind, curves), "characteristic_curves")

length_result <- function(keep) {
    models <- cosine[keep, , drop = FALSE]
    people <- eligible[keep, ]
    controls <- adjustments$age_sex_adjusted
    if (identical(keep, primary)) {
        result <- primary_result
        ci <- subset(intervals$age_sex_adjusted, statistic == "median")
    } else {
        result <- associations(models, people, controls)
        draws <- bootstrap_coefficients(models, people, controls)
        ci <- panel_intervals(draws, ncol(models), list(median = median))
    }
    result$summary$median_ci_lower <- ci$ci_lower
    result$summary$median_ci_upper <- ci$ci_upper
    result
}

minimum_lengths <- 2:20
counts <- vapply(minimum_lengths, function(k) sum(lengths >= k), integer(1))
first <- !duplicated(counts)
by_length <- lapply(minimum_lengths[first], function(k) length_result(lengths >= k))
for (table in c("estimates", "summary")) {
    rows <- Map(function(k, i) data.frame(minimum_speech_length = k, by_length[[i]][[table]]),
                minimum_lengths, match(counts, counts[first]))
    save_results(do.call(rbind, rows), paste0("speech_length_", table))
}
