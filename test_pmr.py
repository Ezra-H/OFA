import copy
# n = int(input())
# A = input().split(' ')
# A = [int(a) for a in A]
# # n = 4
# # A = [5,0,3,1]
#
# res = []
# A_set = set(A)
# ort_mex = 0
# min_A = min(A)
# max_A = max(A)
# for i in range(max_A):
#     if i not in A_set:
#         ort_mex = i
#         break
#
# large_than_min = ort_mex > min_A
# if not large_than_min:
#     res = [ort_mex] * len(A)
# else:
#     for i in range(n):
#         if A[i] > ort_mex:
#             res.append(ort_mex)
#         else:
#             B = A[:i] + A[i+1:]
#             for j in range(0, A[i]+1):
#                 if j not in B:
#                     res.append(j)
#                     break
#
# print(" ".join(map(str, res)))
#
# n,m,k = 3,5,1
#
# c = [2,1,2,3,2]
# a = [9,6,2,1,7]
# b = [1,3,0,5,2]
#
# global_score = []
#
# def run_step(current_score, begin_city, current_day):
#     if current_day >= m:
#         global_score.append(current_score)
#     else:
#         if c[current_day] == begin_city:
#             current_score += a[current_day]
#             run_step(current_score, c[current_day], current_day + 1)
#         else:
#             run_step(current_score, begin_city, current_day + 1)
#             current_score += b[current_day]
#             run_step(current_score, c[current_day], current_day + 1)
#
# run_step(0, k, 0)
# print(max(global_score))

import math
# n, c, r = list(map(int, input().split()))
# id2score = dict()
# for i in range(n):
#     i, score = list(map(int, input().split()))
#     id2score[i] = score
if __name__ == '__main__':
    n,c,r = 5, 3, 1
    id2score = {1:650, 2:640, 3:630, 4:620, 5:610}
    id_and_score = sorted(list(id2score.items()), key=lambda x: x[1], reverse=True)
    ids, scores = zip(*id_and_score)
    print(ids, scores)
    
    m = math.ceil(n/c)
    n_sign_round = math.ceil(n/m)  # 向上取整 多多那轮也包含了
    duoduo_idx = r // m
    
    n_situation = m**(n_sign_round - 1)
    
    # 还要考虑多多排最后一轮的情况。
    all_student_pack = []
    for i in range(0, n, m):
        all_student_pack.append(scores[i:i+m])
    
    all_student_pack_average = []
    for i in range(len(all_student_pack)):
        all_student_pack_average.append(sum(all_student_pack[i])/len(all_student_pack[i]))
        
    is_last_full = len(all_student_pack[-1]) == n_sign_round
    duoduo_in_last = duoduo_idx == len(all_student_pack_average) - 1
    
    # average = (sum(all_student_pack_average) - all_student_pack_average[duoduo_idx] + scores[r]) / n_situation  + \
    #           (~(duoduo_in_last or is_last_full)) * (sum(all_student_pack_average[:-1]) - all_student_pack_average[duoduo_idx] + scores[r]) / (n_situation - 1)
    # print(average)
    
    average = ((sum(all_student_pack_average) - all_student_pack_average[duoduo_idx] + scores[r]) / len(all_student_pack_average) * len(all_student_pack[-1]) + \
               (sum(all_student_pack_average[:-1]) - all_student_pack_average[duoduo_idx] + scores[r]) / (len(all_student_pack_average)-1) * (m-len(all_student_pack[-1]))) / m
print(average)

    
