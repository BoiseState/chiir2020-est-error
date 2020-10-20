
library(tidyverse)

partition_ratings = function(ratings, nparts=5) {
    nRows = nrow(ratings)
    # shuffle the row number
    index = sample(nRows)
    # equally divide nRows into nparts
    partSizes = (nRows %/% nparts) * rep(1, nparts)
    # equally divide the remainder and add to the first "remainder" parts
    remainder = nRows %% nparts
    if ( remainder > 0) {
        partSizes[1:remainder] = partSizes[1:remainder] + 1
    }
    partInd = data_frame(part = 1:nparts, rowNumber=list(NULL))
    startInd = 1
    for (i in 1:nparts) {
        endInd = startInd + partSizes[[i]] - 1
        partInd[[i, "rowNumber"]] = index[startInd:endInd]
        startInd = endInd + 1
    }
    partInd %>% unnest(rowNumber)
}

partition_users = function(ratings, nparts=5, holdout=5) {
    colnames(ratings) = c("userId", "itemId")
    # sample test set, for each user, select "holdout" items
    testSet = ratings %>%
        mutate(rowNumber=row_number()) %>%
        group_by(userId) %>%
        mutate(index = sample(n())) %>%
        filter(index <= holdout) %>%
        ungroup() %>%
        select(-index)
    uniqueUsers = unique(ratings$userId)
    # shuffle the userId
    uniqueUsers = sample(uniqueUsers)
    nusers = length(uniqueUsers)
    # partition users
    userPartitions = data_frame(part = 1:nparts, userId=list(NULL))
    partSizes = (nusers %/% nparts) * rep(1, nparts)
    # equally divide the remainder and add to the first "remainder" parts
    remainder = nusers %% nparts
    if ( remainder > 0) {
        partSizes[1:remainder] = partSizes[1:remainder] + 1
    }
    startInd = 1
    for (i in 1:nparts) {
        endInd = startInd + partSizes[[i]] - 1
        userPartitions[[i, "userId"]] = uniqueUsers[startInd:endInd]
        startInd = endInd + 1
    }
    userPartitions = userPartitions %>% unnest(userId)
    # join userId partition with users' items return rowNumber for reference
    suppressMessages(userPartitions %>% 
        inner_join(testSet) %>%
        select(part, rowNumber))
}

recommend_oracle = function(candidates, groundtruth, observation=NULL, topN) {
    suppressMessages(candidates %>%
        left_join(groundtruth %>% mutate(score = 1)) %>% # the score means prediction
        mutate(score = ifelse(is.na(score), 0, score)) %>%
        group_by(userId) %>%
        mutate(rank = rank(-score, ties.method = "first")) %>%
        ungroup() %>%
        arrange(userId, rank) %>%
        filter(rank <= topN))
}

recommend_popular = function(candidates, groundtruth=NULL, observation, topN) {
    # prediction
    popularScore = observation %>%
        group_by(itemId) %>%
        summarize(n = n()) %>%
        mutate(score = n / max(n)) %>%
        select(-n)
    # join with predicted scores
    suppressMessages(candidates %>%
        left_join(popularScore) %>%
        group_by(userId) %>%
        mutate(rank = rank(-score, ties.method = "first")) %>%
        ungroup() %>%
        arrange(userId, rank) %>%
        filter(rank <= topN))
}

recommend_random = function(candidates, groundtruth=NULL, observation=NULL, topN) {
    candidates %>%
        group_by(userId) %>%
        mutate(score = runif(n()),
               rank = rank(-score, ties.method = "first")) %>%
        ungroup() %>%
        arrange(userId, rank) %>%
        filter(rank <= topN)
}

# ideal: all rated items for each user. userId-itemId-rel
# recommendations: userId-itemId-score-rank
compute_ndcg = function(recommendations, ideal, topN) {
    dcg = suppressMessages(recommendations %>%
        left_join(ideal) %>%
        mutate(rel=ifelse(is.na(rel), 0, rel),
               dg=ifelse(rank==1, rel, rel / log2(rank))) %>%
        group_by(userId) %>%
        summarize(dcg = sum(dg)) %>%
        ungroup())
    idcg = ideal %>%
        group_by(userId) %>%
        mutate(rank = row_number()) %>%
        ungroup() %>%
        filter(rank <= topN) %>%
        group_by(userId) %>%
        mutate(idg=ifelse(rank==1, rel, rel / log2(rank))) %>%
        summarize(idcg = sum(idg)) %>%
        ungroup()
    
    suppressMessages(dcg %>%
        inner_join(idcg) %>%
        mutate(ndcg = dcg / idcg) %>%
        select(userId, ndcg))
}

compute_precision = function(recommendations, ideal, topN) {
    suppressMessages(recommendations %>%
        filter(rank <= topN) %>%
        left_join(ideal) %>%
        mutate(rel=ifelse(is.na(rel), 0, rel)) %>%
        group_by(userId) %>%
        summarize(precision=mean(rel)) %>%
        ungroup())
}

compute_recall = function(recommendations, ideal, topN) {
    tp = suppressMessages(recommendations %>%
        filter(rank <= topN) %>%
        left_join(ideal) %>%
        group_by(userId) %>%
        summarize(tp = sum(rel, na.rm = TRUE)) %>%
        ungroup())
    relevanceCount = ideal %>%
        group_by(userId) %>%
        summarize(relCount = n()) %>%
        ungroup()
    suppressMessages(tp %>%
        inner_join(relevanceCount) %>%
        transmute(userId=userId,
                  recall=tp / relCount))
}

compute_reciprocal_rank = function(recommendations, ideal, topN) {
    validUserRR = suppressMessages(recommendations %>%
        filter(rank <= topN) %>%
        left_join(ideal) %>%
        filter(!is.na(rel)) %>% # lose users
        group_by(userId) %>%
        summarize(rankFirstRel=min(rank)) %>%
        ungroup() %>%
        transmute(userId=userId,
                  reciprocal.rank=1 / rankFirstRel))
    
    suppressMessages(recommendations %>%
        select(userId) %>%
        distinct() %>%
        left_join(validUserRR))
}

compute_average_precision = function(recommendations, ideal, topN) {
    validUserAP = suppressMessages(recommendations %>%
        filter(rank <= topN) %>%
        left_join(ideal) %>%
        mutate(rel=ifelse(is.na(rel), 0, rel)) %>%
        group_by(userId) %>%
        arrange(rank) %>%
        mutate(cumsumRel = cumsum(rel),
               precisionAtRank = cumsumRel / rank) %>%
        ungroup() %>%
        filter(rel != 0) %>% # lose users
        group_by(userId) %>%
        summarize(avg.precision=mean(precisionAtRank)) %>%
        ungroup())
    
    suppressMessages(recommendations %>%
        select(userId) %>%
        distinct() %>%
        left_join(validUserAP))
}

compute_hit_rate = function(recommendations, ideal, topN) {
    suppressMessages(recommendations %>%
        filter(rank <= topN) %>%
        left_join(ideal) %>%
        group_by(userId) %>%
        summarize(hit=any(!is.na(rel))) %>%
        ungroup())
}

generate_preference = function(generators=list(), parameters=list()) {
    genNames = names(generators)
    if (is.null(genNames)) {
        genNames = 1:length(generators)
    }
    results = list()
    # feed each generator with each set of inpute parameters
    for (i in 1:length(generators)) {
        generator = generators[[i]]
        parameter = parameters[[i]]
        result = do.call(generator, parameter)
        results[[genNames[i]]] = result
    }
    results
}

sample_obseravation = function(preferences=list(), samplingStrategies=list(), parameters=list()) {
    # get preference names
    prefNames = names(preferences)
    npref = length(preferences)
    if (is.null(prefNames)) {
        prefNames = 1:npref
    }
    # get sampler names
    samplerNames = names(samplingStrategies)
    nsampler = length(samplingStrategies)
    if (is.null(samplerNames)) {
        samplerNames = 1 : nsampler
    }
    
    results = list()
    for (i in 1:npref) {
        prefName = prefNames[i]
        pref = preferences[[i]]
        for (j in 1:nsampler) {
            resultName = paste(prefName, samplerNames[j], sep="-")
            sampler = samplingStrategies[[j]]
            result = do.call(sampler, c(list(pref), parameters))
            results[[resultName]] = result
        }
    }
    results
}

build_recommender = function(preferences=list(), observations=list(),
                             partitionStrategy=partition_users, partitionParam=list(),
                             recommenders=list(), topN=10,
                             evalMetrics=list()) {
    obsNames = names(observations)
    if (is.null(obsNames)) {
        obsNames = 1:length(observations)
    }
    obsNamesList = split(obsNames, obsNames)
    
    recNames = names(recommenders)
    if (is.null(recNames)) {
        recNames = 1:length(recommenders)
    }
    recNamesList = split(recNames, recNames)
    
    map_dfr(obsNamesList, function(obsName) {
        # get the name of preference model
        prefNames = strsplit(obsName, split = "-")[[1]][[1]]
        # get preference
        pref = preferences[[prefNames]]
        # get observation data
        obs = observations[[obsName]]
        # all itemIds in obs
        allItems = unique(obs$itemId)
        # partition current observation
        testSet = do.call(partitionStrategy, c(list(obs), partitionParam))
        nparts = unique(testSet$part)
        # convert vector to list
        npartsList = split(nparts, nparts)
        # build recommenders for each partition
        map_dfr(npartsList, function(partNum){
            # get data in current partition
            testPartRowNumber = testSet %>% filter(part == partNum)
            test = obs %>%
                filter(row_number() %in% testPartRowNumber$rowNumber) 
            train = suppressMessages(obs %>%
                anti_join(test)) 
            candidates = suppressMessages(test %>%
                group_by(userId) %>%
                summarize(itemId=list(allItems)) %>%
                ungroup() %>%
                unnest(itemId) %>%
                anti_join(train))
            idealRelObv = obs %>%
                filter(userId %in% unique(test$userId)) %>%
                mutate(rel=1)
            idealRelGroundTruth = pref %>%
                filter(userId %in% unique(test$userId)) %>%
                mutate(rel=1)
            
            # build recommendations and compute metrics
            map_dfr(recNamesList, function(recName){
                recommender = recommenders[[recName]]
                recommendation = recommender(candidates, pref, obs, topN)
                obsMetricsList = map(evalMetrics, do.call, list(recommendation, idealRelObv, topN))
                # combine into dataframe
                obsMetrics = reduce(obsMetricsList, function(x1, x2){
                    suppressMessages(inner_join(x1, x2))
                })
                gtMetricsList = map(evalMetrics, do.call, list(recommendation, idealRelGroundTruth, topN))
                gtMetrics = reduce(gtMetricsList, function(x1, x2){
                    suppressMessages(inner_join(x1, x2))
                })
                bind_rows(observation=obsMetrics, groundtruth=gtMetrics, .id="evaluation")       
            }, .id="algorithm")
            
        }, .id="part")
        
    }, .id="strategy")
}

