# -*- coding: utf-8 -*-


GENRE2LABELMAP = {
    "Action":"1","Comedy":"2","Animation":"3","Children's":"3",
    "Horror":"4","Thriller":"4","Film-Noir":"4","Documentary":"5",
    "Drama":"6","War":"7","Western":"7","Crime":"7","Adventure":"7",
    "Musical":"8","Fantasy":"8","Romance":"8","Mystery":"9","Sci-Fi":"9",
}

ALL_LABELS = ["1","2","3","4","5","6","7","8","9"]

REDIS_USER_TRACE = "user_trace"
REDIS_HOT_MOVIES = "hot_movies"
REDIS_COLD_MOVIES = "cold_movies"


REDIS_USER_LIKE_GENRE_WEIGHT = "user_like_genre_weight"
REDIS_USER_HISTORY_REC_MOVIES = "user_history_rec_movies"
REDIS_USER_HISTORY_CLICK_MOVIES = "user_history_click_movies"
REDIS_USER_HISTORY_CLICK_GENRES = "user_history_click_genres"


REDIS_CURRENT_WEIGHTS = "current_weights"
REDIS_LONG_TERM_INTEREST_50 = "long_term_interest_50"
REDIS_MIDDLE_TERM_INTEREST_20 = "middle_term_interest_20"
REDIS_SHORT_TERM_INTEREST_10 = "short_term_interest_10"
REDIS_BALANCE_TERM_INTEREST_10_20_50 = "balance_term_interest_10_20_50"
REDIS_FINISH_WEIGHTS = "finish_weights"


REDIS_MOVIE_INFO = "movie_info"



"""召回配置"""
REDIS_MATCH_USER_RECENT_HISTORY_CLICK_MOVIE_TRACE = "match_user_recent_history_click_movie_trace"




#INIT_USER_LIKE_GENRE_WEIGHT = {"1":1,"2":1,"3":1,"4":1,"5":1,"6":1,"7":1,"8":1,"9":1}

"""初始化 各个类别的权重"""
INIT_WEIGHT = {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2, "9": 2}


USER_ADD_GENRE_WEIGHT = 0.2

INIT_USER_LIKE_GENRE_PRIORITY = {"2":0.138,"1":0.131,"6":0.125,"7":0.118,"4":0.111,"3":0.104,"9":0.097,"8":0.090,"5":0.083}

