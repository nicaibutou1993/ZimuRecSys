"""
求差集 并保持 集合1 位置不变
"""

def diff_and_sort(list1, list2):
    # output = []
    # if len(list1) > 0:
    #     o = {l: i for i, l in enumerate(list1)}
    #
    #     diff = set(list1) - set(list2)
    #
    #     sorted_data = sorted([(o.get(d), d) for d in diff], key=lambda x: x[0])
    #
    #     output = [d for i, d in sorted_data]

    output = [i for i in list1 if i not in list2]

    return output

#hot_movies = [i for i in hot_movies if i not in history_movies]
if __name__ == '__main__':
    a = ['34', '1197', '1136', '2804', '1293', '25', '50', '515', '1674', '2289', '3751', '3408',
         '3753', '3868', '551',
         '2064',
         '3359', '1285', '2908', '1171', '1272', '41',
         '3250', '1956', '852']

    b = ['1148', '1', '50', '10', '1004', '13']

    sort = diff_and_sort(a, b)
    print(sort)








