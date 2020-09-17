import matplotlib.pyplot as plt
import numpy as np
import os

'''
x_data = np.linspace(1, 10, 10)
y_data = np.array([0.9345, 0.9577, 0.9631, 0.9406, 0.9588, 0.9455, 0.9717, 0.9614, 0.9608, 0.9478])

plt.title("Result of left lung segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')
plt.xticks(x_data,('P000333562', 'P00035575', 'P00035756', 'P00037427','P00038535',
                   'P00038719', 'P00041997-329071', 'P00042280', 'P00042613', 'P00052478'),
           rotation = 10)
#plt.yticks(np.arange(y_data.min(), 1.0, 0.01))
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data, label='value', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.85), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.80), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.75), color = 'blue')

plt.legend()
plt.show()
'''

'''
x_data = np.linspace(1, 25, 25)
y_data = np.array([0.6187, 0.7798, 0.8051, 0.8229, 0.7743,
                   0.7673, 0.8724, 0.8501, 0.7613, 0.7681,
                   0.7969, 0.7977, 0.7460, 0.7692, 0.8130,
                   0.7264, 0.7782, 0.6596, 0.8297, 0.8931,
                   0.7947, 0.8482, 0.8016, 0.7829, 0.7956])

plt.title("Result of left eye segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')
plt.xticks(x_data,('BYS060', 'BYS061', 'BYS062','BYS063', 'BYS064',
                   'BYS065', 'BYS066', 'BYS067', 'BYS068', 'BYS069',
                   'BYS070', 'BYS071', 'BYS072', 'P000224211', 'P00053705',
                   'P00058666', 'P00059459', 'P00060822', 'P00062317', 'P00063184',
                   'P00065248', 'P00065279', 'P00065466', 'P00066058', 'P00068478',
                   ),
           rotation = 15)
#plt.yticks(np.arange(y_data.min(), 1.0, 0.01))
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data, label='value', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.50), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.45), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.4), color = 'blue')

plt.legend()
plt.show()
'''

'''
x_data = np.linspace(1, 22, 22)
y_data = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])

plt.title("Result of left hippocampus segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')
plt.xticks(x_data,('R1278362', 'R1446516', 'R1537736', 'R1607043', 'R1906544',
                   'R2083827_1', 'R2114916', 'R2160318', 'R2371825', 'R349781',
                   'geservice_2', 'R2583029', 'R41094222', 'R41225052', 'R41232261',
                   'R41236186', 'R41256725', 'R41273424', 'R41281897', 'R41291374',
                    'R41301352', 'R41311267'),
           rotation = 15)
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data, label='value', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.50), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.45), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color = 'blue')

plt.legend()
plt.show()
'''

'''
x_data = np.linspace(1, 21, 21)
y_data_brain_stem = np.array([0.7472, 0.7967, 0.6248, 0.7651, 0.616,
                              0.818, 0.8649, 0.8071, 0.759, 0.8487,
                              0.83, 0.7241, 0.7433, 0.7869, 0.819,
                              0.6839, 0.6469, 0.6499, 0.7655, 0.6755,
                              0.657])
y_data_mandible = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])
y_data_parotid_l = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])
y_data_parotid_r = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])

plt.title("Result of brainstem segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')

plt.xticks(x_data,('301709_296179hn', '325795hn', '340070', '53', 'P00009831hn',
                   'P00016493hn', 'P00018267hn', 'P00019760hn', 'P00020569hn', 'P00021288_319285hn',
                   'P00021600-319382hn', 'P00021600hn', 'P00023575hn', 'P00025483hn', 'P00026377hn',
                   'P00029379hn', 'P00030036hn', 'Wang rong kang', 'Zheng da heng', 'Zhou gui feng',
                    'Zhou jin fu', 'Zhou lv bin', 'Zhou xiao ying', 'Zhu ji ye', 'Zhu lai fa',
                   'Zhu ren dong', 'chenjianhai', 'dingmin', 'donglanxiang', 'gaoyuxiu',
                    'gongguihua', 'jiang hong', 'jianghongde', 'jiangzouzhi', 'songguiyue',
                   'songyanfei', 'suiyongzeng', 'sunguihu', 'yuzhihui'),
           rotation = 35)
plt.xticks(x_data,('pid01', 'pid02', 'pid03', 'pid04', 'pid05', 'pid06', 'pid07', 'pid08', 'pid09', 'pid10',
                   'pid11', 'pid12', 'pid13', 'pid14', 'pid15', 'pid16', 'pid17', 'pid18', 'pid19', 'pid20',
                   'pid21'),
           rotation = 35)
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data_brain_stem, label='brainstem', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data_brain_stem):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data_brain_stem)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data_brain_stem)), xy=(1, 0.50), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data_brain_stem)), xy=(1, 0.45), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color = 'blue')

plt.legend()
plt.show()
'''
'''
x_data = np.linspace(1, 25, 25)
y_data_brain_stem = np.array([0.8606, 0.8706, 0.8779, 0.7645, 0.7638,
                              0.7871, 0.878, 0.8652, 0.8274, 0.8274,
                              0.711, 0.836, 0.8449, 0.8398, 0.797,
                              0.8738, 0.8906, 0.8124, 0.8629, 0.8549,
                              0.7666, 0.827, 0.7911, 0.7984, 0.8202])
y_data_mandible = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])
y_data_parotid_l = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])
y_data_parotid_r = np.array([0.7222, 0.7, 0.8160, 0.7531, 0.6524,
                   0.6666, 0.7157, 0.7687, 0.7116, 0.7672,
                   0.7181, 0.7816, 0.8207, 0.7029, 0.7687,
                   0.7804, 0.7075, 0.7414, 0.7964, 0.7055,
                   0.6804, 0.8096])

plt.title("Result of mandible segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')
plt.xticks(x_data,('pid01', 'pid02', 'pid03', 'pid04', 'pid05', 'pid06', 'pid07', 'pid08', 'pid09', 'pid10',
                   'pid11', 'pid12', 'pid13', 'pid14', 'pid15', 'pid16', 'pid17', 'pid18', 'pid19', 'pid20',
                   'pid21', 'pid22','pid23','pid24','pid25'),
           rotation = 35)
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data_brain_stem, label='mandible', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data_brain_stem):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data_brain_stem)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data_brain_stem)), xy=(1, 0.50), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data_brain_stem)), xy=(1, 0.45), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color = 'blue')

plt.legend()
plt.show()
'''

'''
x_data = np.linspace(1, 20, 20)
y_data = np.array([0.8504, 0.8443, 0.8261, 0.7458, 0.8331,
                   0.8509, 0.7016, 0.7599, 0.7876, 0.7899,
                   0.7793, 0.7481, 0.7473, 0.7992, 0.8376,
                   0.7383, 0.8224, 0.7913, 0.7153, 0.7153])

plt.title("Result of left parotid segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')
plt.xticks(x_data,('pid01', 'pid02', 'pid03', 'pid04', 'pid05', 'pid06', 'pid07', 'pid08', 'pid09', 'pid10',
                   'pid11', 'pid12', 'pid13', 'pid14', 'pid15', 'pid16', 'pid17', 'pid18', 'pid19', 'pid20'),
           rotation = 35)
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data, label='left parotid', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.50), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.45), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color = 'blue')

plt.legend()
plt.show()
'''
'''
x_data = np.linspace(1, 23, 23)
y_data = np.array([0.8302, 0.7975, 0.8409, 0.772, 0.8245,
                   0.8562, 0.7279, 0.7279, 0.7339, 0.7438,
                   0.7816, 0.8241, 0.7986, 0.7277, 0.8198,
                   0.7086, 0.8118, 0.7354, 0.7354, 0.7812,
                   0.7914, 0.7474, 0.7474])

plt.title("Result of right parotid segmentation with AI")
plt.xlabel('patient id')
plt.ylabel('dice value')
plt.xticks(x_data,('pid01', 'pid02', 'pid03', 'pid04', 'pid05', 'pid06', 'pid07', 'pid08', 'pid09', 'pid10',
                   'pid11', 'pid12', 'pid13', 'pid14', 'pid15', 'pid16', 'pid17', 'pid18', 'pid19', 'pid20',
                   'pid21', 'pid22', 'pid23'),
           rotation = 35)
plt.yticks(np.arange(0, 1.0, 0.1))
plt.ylim(0, 1)

l1 = plt.plot(x_data, y_data, label='right parotid', linewidth=2, color='r', marker='o', markerfacecolor='k', markersize=5)

for a, b in zip(x_data, y_data):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

mean_val = np.mean(y_data)
#mean_data = np.full(np.shape(x_data), mean_val)
#l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.50), color = 'red')
plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.45), color = 'green')
plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color = 'blue')

plt.legend()
plt.show()
'''

def result_plot(oar_name, result_image_full_path, dice_value_list, pid_list):
    assert len(dice_value_list) == len(pid_list)

    dir_name = os.path.dirname(result_image_full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    nums = len(dice_value_list)
    x_data = np.linspace(1, nums, nums)
    y_data = np.zeros([nums], dtype=np.float32)
    for i in range(nums):
        y_data[i] = dice_value_list[i]

    ticks = '('
    for i in range(nums - 1):
        ticks = ticks + '\'' + '%s' % (pid_list[i]) + '\'' + ', '
    ticks = ticks + '\'%d' % (pid_list[nums - 1]) + '\'' + ')'

    plt.figure(figsize=(1928 / 100, 1048 / 100))
    plt.title('Result of %s segmentation with AI' % (oar_name))
    plt.xlabel('patient id')
    plt.ylabel('dice value')
    plt.xticks(x_data, eval(ticks), rotation=35)
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.ylim(0, 1)

    l1 = plt.plot(x_data, y_data, label=oar_name, linewidth=2, color='r', marker='o', markerfacecolor='k',
                  markersize=5)

    for a, b in zip(x_data, y_data):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    mean_val = np.mean(y_data)
    # mean_data = np.full(np.shape(x_data), mean_val)
    # l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

    plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.50), color='red')
    plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.45), color='green')
    plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color='blue')

    plt.legend()
    #plt.show()
    plt.savefig(result_image_full_path)

def result_plot1(oar_name1, oar_name2, result_image_full_path, dice_value_list1, dice_value_list2, pid_list):
    assert len(dice_value_list1) == len(pid_list)
    assert len(dice_value_list2) == len(pid_list)

    dir_name = os.path.dirname(result_image_full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    nums = len(dice_value_list1)
    x_data = np.linspace(1, nums, nums)
    y_data = np.zeros([nums], dtype=np.float32)
    for i in range(nums):
        y_data[i] = dice_value_list1[i]

    nums2 = len(dice_value_list1)
    y_data2 = np.zeros([nums2], dtype=np.float32)
    for i in range(nums2):
        y_data2[i] = dice_value_list2[i]

    ticks = '('
    for i in range(nums - 1):
        ticks = ticks + '\'' + '%s' % (pid_list[i]) + '\'' + ', '
    ticks = ticks + '\'%d' %(pid_list[nums - 1]) + '\'' + ')'

    plt.figure(figsize=(1928 / 100, 1048 / 100))
    plt.title('Result of %s segmentation with AI' % ('hippocampus'))
    plt.xlabel('id')
    plt.ylabel('dice value')
    plt.xticks(x_data, eval(ticks), rotation=35)
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.ylim(0, 1)

    l1 = plt.plot(x_data, y_data, label=oar_name1, linewidth=2, color='r', marker='o', markerfacecolor='k',
                  markersize=5)
    l2 = plt.plot(x_data, y_data2, label=oar_name2, linewidth=2, color='g', marker='*', markerfacecolor='k',
                  markersize=5)

    for a, b in zip(x_data, y_data):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    for a, b in zip(x_data, y_data2):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    mean_val = np.mean(y_data)
    mean_val2 = np.mean(y_data2)
    # mean_data = np.full(np.shape(x_data), mean_val)
    # l2 = plt.plot(x_data, mean_data, label='mean value', color='green',linestyle='--')

    plt.annotate('max value:%.4f ' % (np.max(y_data)), xy=(1, 0.50), color='red')
    plt.annotate('min value:%.4f ' % (np.min(y_data)), xy=(1, 0.45), color='red')
    plt.annotate('mean value:%.4f ' % (mean_val), xy=(1, 0.40), color='red')

    plt.annotate('max value:%.4f ' % (np.max(y_data2)), xy=(4, 0.50), color='green')
    plt.annotate('min value:%.4f ' % (np.min(y_data2)), xy=(4, 0.45), color='green')
    plt.annotate('mean value:%.4f ' % (mean_val2), xy=(4, 0.40), color='green')

    plt.legend()
    #plt.show()
    plt.savefig(result_image_full_path)

oarname1 = 'hippo-L'
oarname2 = 'hippo-R'
# dice_value_list1 = [0.8,0.86,0.85,0.86,0.85,0.81,0.85,0.88,0.88,0.85,0.84,0.81,0.86,0.85,0.84,0.86,0.85,0.86,0.88,0.86,0.81,0.86,0.85,0.81,0.83,0.87]
# dice_value_list2 = [0.84,0.82,0.82,0.85,0.88,0.86,0.85,0.85,0.86,0.86,0.82,0.82,0.88,0.88,0.88,0.88,0.89,0.88,0.85,0.87,0.84,0.89,0.86,0.8,0.86,0.89]
# pid_list = [1,9,16,27,36,49,54,57,66,71,75,82,88,97,101,110,117,119,129,137,142,146,159,166,186,187]

dice_value_list1 = [0.857
,0.843
,0.847
,0.803
,0.85
,0.894
,0.862
,0.879
,0.879
,0.88
,0.872
,0.845
,0.818
,0.876
,0.9
,0.857
,0.889
,0.881
,0.873
,0.908
,0.882
,0.81
,0.892
,0.839
,0.862
,0.833
,0.863
,0.871
,0.875
,0.75]
dice_value_list2 = [0.85
,0.806
,0.844
,0.788
,0.877
,0.884
,0.874
,0.843
,0.852
,0.87
,0.884
,0.802
,0.823
,0.89
,0.903
,0.887
,0.873
,0.884
,0.882
,0.872
,0.86
,0.87
,0.892
,0.858
,0.87
,0.834
,0.813
,0.852
,0.878
,0.82]
pid_list = [1,9,16,25,27,36,49,54,57,66,71,75,82,88,97,101,110,117,119,129,137,142,146,152,159,161,166,186,187, 205]

result_image_full_path = 'D:/result1.png'
result_plot1(oarname1, oarname2, result_image_full_path, dice_value_list1, dice_value_list2, pid_list)

result_image_full_path = 'D:/result2.png'
result_plot(oarname1, result_image_full_path, dice_value_list1, pid_list)

result_image_full_path = 'D:/result3.png'
result_plot(oarname2, result_image_full_path, dice_value_list2, pid_list)