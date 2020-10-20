library(tidyverse)
library(assertthat)

summarize_data_set = function(df) {
    colNames = colnames(df)
    users = df %>% 
        select_at(1) %>%
        distinct() %>%
        nrow()
    items = df %>% 
        select_at(2) %>%
        distinct() %>%
        nrow()
    pairs = df %>%
        distinct() %>%
        nrow()
    density = pairs / users / items
    
    stats = data.frame(user=users,
               item=items,
               pair=pairs,
               density=density)
    print(stats)
    
    
    
    userProfiles = df %>%
        group_by_at(1) %>%
        summarize(Profile = n()) %>%
        ungroup() %>%
        select_at(2) 
    
    itemPopularity = df %>%
        group_by_at(2) %>%
        summarize(Popularity = n()) %>%
        ungroup() %>%
        select_at(2)
    
    print(summary(userProfiles))
    print(summary(itemPopularity))
    
    print(ggplot(userProfiles) +
        aes(x=Profile) +
        geom_histogram() +
        xlab("Profile Size") +
        ylab("# of Users"))
    
    itemPopRank = itemPopularity %>%
        mutate(Rank=rank(-Popularity, ties.method = 'min')) %>%
        distinct() %>%
        arrange(Rank)
    print(ggplot(itemPopRank) +
        aes(x=Rank, y=Popularity) +
        geom_point() +
        scale_x_log10() +
	scale_y_log10())

    popularityDist = itemPopularity %>%
	group_by_at(1) %>%
	summarize(count = n()) %>%
	ungroup()
    print(ggplot(popularityDist) + 
	aes(x=Popularity, y=count) +
	geom_point() +
	scale_x_log10() +
	scale_y_log10())    
}

summarize_data_sets = function(...) {
    args = list(...)
    # combine all the data
    data = map_dfr(args, function(df) {
        colnames(df) = c("userId", "itemId")
        df
    }, .id = "DataSet")
    
    # compute statistics of each data set
    stats = data %>%
        distinct() %>%
        group_by(DataSet) %>%
        summarize(nuser = n_distinct(userId),
                  nitem = n_distinct(itemId),
                  pairs = n(),
                  density = pairs / nuser / nitem) %>%
        ungroup()
    
    print(as.data.frame(stats))
    
    # compute user profiles and item popularities
    userProfiles = data %>%
        group_by(DataSet, userId) %>%
        summarize(Profile = n()) %>%
        ungroup()

    itemPopularity = data %>%
        group_by(DataSet, itemId) %>%
        summarize(Popularity = n()) %>%
        ungroup()
    
    message("Profile Stats")
    print(userProfiles %>%
        spread(DataSet, Profile) %>%
        select(-userId) %>%
        summary())
    
    message("Popularity Stats")
    print(itemPopularity %>%
        spread(DataSet, Popularity) %>%
        select(-itemId) %>%
        summary())

    # histogram of user profiles
    print((ggplot(userProfiles) +
        aes(x=Profile) +
        geom_histogram() +
        xlab("Profile Size") +
        ylab("# of Users") +
        facet_wrap(~DataSet,scales="free")))
    
    # compute popularity rank
    itemPopularity = itemPopularity %>%
        group_by(DataSet) %>%
        mutate(Rank=rank(-Popularity, ties.method = 'min')) %>%
        ungroup() %>%
        select(-itemId) %>%
        arrange(DataSet, Rank)
    
    print(ggplot(itemPopularity) +
        aes(x=Rank, y=Popularity) +
        geom_point() +
        scale_x_log10() +
        scale_y_log10() +
        facet_wrap(~DataSet,scales="free"))
    
    itemPopularity = itemPopularity %>%
        group_by(DataSet, Popularity) %>%
        summarize(Count = n()) %>%
        ungroup()
    
    # scatter plot of frequency vs item popularity
    print(ggplot(itemPopularity) + 
        aes(x=Popularity, y=Count) +
        geom_point() +
        scale_x_log10() +
        scale_y_log10() +
        facet_wrap(~DataSet,scales="free"))
}

generate_uniform = function(nusers, nitems, frac) {
    lambda = frac * nitems
    # get users and their item counts
    users = data_frame(userId=1:nusers) %>%
        mutate(itemCount=rpois(n(), lambda))
    # now we select items, unnest, and be done!
    users %>% rowwise() %>%
        mutate(itemId=list(sample(nitems, itemCount))) %>%
        select(-itemCount) %>%
        unnest(itemId)
}

generate_IBP = function(nusers, avgItems) {
    start = Sys.time()
    # keep track of items in use
    nitems = 0
    items = numeric()
    # user info - data frame with nested column for items
    ratings = data_frame(userId = 1:nusers, ic=0, items=list(NULL))
    for (u in 1:nusers) {
        # pick from previously-used items
        pick = rbinom(nitems, 1, items / u) > 0
        reused = (1:nitems)[pick]
        # update item counts
        items[pick] = items[pick] + 1
        # pick new items
        nnew = rpois(1, avgItems/u)
        new = (1:nnew) + nitems
        # add new items to used
        if (nnew > 0) {
            items = c(items, rep(1, nnew))
            nitems = nitems + nnew
            ratings[[u,"items"]] = c(reused, new)
        } else {
            ratings[[u,"items"]] = reused
        }
        assert_that(are_equal(length(items), nitems))
    }
    results = ratings %>% unnest(items) %>% select(userId, itemId=items)
    
    end = Sys.time()
    message("sampled ", nrow(results), " interactions for ", nusers, " users in ", end - start, " seconds")
    results
}

generate_IBP_old = function(nusers, nitems, frac) {
    start = Sys.time()
    alpha = nitems * frac
    historyRatings = setNames(data.frame(matrix(ncol = 3, nrow = 0)), c('userId', 'itemId', 'ratings'))
    nConsumedItems = 0
    
    userItemPairs = map_dfr(1:nusers, function(i) {
        samplePopularDishes = historyRatings %>%
        group_by(itemId) %>%
        summarize(prob = n()) %>%
        ungroup() %>%
        mutate(prob = prob / i) %>%
        transmute(userId = i,
                  itemId = itemId,
                  ratings = rbinom(nConsumedItems, size = 1, prob = prob)) %>%
        filter(ratings != 0)

        sampleNewDishes = setNames(data.frame(matrix(ncol = 3, nrow = 0)), c('userId', 'itemId', 'ratings'))
	nNewDishes = rpois(1, alpha / i)
	if (nNewDishes > 0) {
		nCurrentDishes = nConsumedItems + nNewDishes
        	sampleNewDishes = data_frame(userId = i, itemId = (nConsumedItems + 1) : nCurrentDishes, ratings = 1)
		assign("nConsumedItems", nCurrentDishes, inherits=TRUE)
	}
        
        userRatings = rbind(samplePopularDishes, sampleNewDishes)

        assign("historyRatings", rbind(historyRatings, userRatings), inherits=TRUE)

        userRatings       
})
    end = Sys.time()
    message("sampled ", nrow(userItemPairs), " interactions for ", nusers, " users in ", end - start, " seconds")
    return(userItemPairs %>% select(-ratings))
}

# sample userId-itemId pairs uniformly
sample_uniform = function(userItemPairs, frac) {
    userItemPairs %>%
        group_by(userId) %>%
        sample_frac(frac) %>%
        ungroup()
}

# sample userId-itemId pairs based on item popularity
sample_popular = function(userItemPairs, frac) {
    popularity = userItemPairs %>%
        group_by(itemId) %>%
        summarize(prob = n()) %>%
        ungroup() %>%
        mutate(prob = prob / sum(prob))
    userItemPairs %>%
        inner_join(popularity) %>%
        group_by(userId) %>%
        sample_frac(size = frac, weight = prob) %>%
        ungroup() %>%
        select(-prob)
}
