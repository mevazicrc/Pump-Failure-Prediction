
import numpy as np
import pickle
from multiprocessing import cpu_count, Pool

def calc_card_derivatives(pos, load):
    nt = int(len(pos) / 2)

    vel = np.diff(pos)
    vel[nt:] = - vel[nt:]

    x1 = pos[:nt - 1] + 0.5 * np.diff(pos[:nt])
    vel1 = np.diff(pos[:nt])
    x2 = pos[2 * nt:nt:-1] + 0.5 * np.diff(pos[2 * nt:nt - 1:-1])
    vel2 = np.diff(pos[2 * nt:nt - 1:-1])

    vel2_interp = np.interp(x1, x2, vel2)
    del_vel = vel2_interp - vel1

    acc = np.diff(vel)

    st_delvel_mean = np.mean(np.abs(del_vel))
    st_delvel_var = np.var(np.abs(del_vel))

    st_vel_var = np.var(np.abs(acc[:nt - 2]))
    st_vel_mean = np.mean(np.abs(acc[:nt - 2]))
    st_acc_var = np.var(acc)
    st_acc_mean = np.mean(acc)

    ups_vel_var = np.var(np.abs(vel1))
    ups_vel_mean = np.mean(np.abs(vel1))
    ups_acc_var = np.var(acc[:nt - 2])
    ups_acc_mean = np.mean(acc[:nt - 2])

    dns_vel_var = np.var(np.abs(vel2))
    dns_vel_mean = np.mean(np.abs(vel2))
    dns_acc_var = np.var(acc[nt - 2:])
    dns_acc_mean = np.mean(acc[nt - 2:])

    # Load derivatives
    n_cut = 20
    pos_u = pos[n_cut:nt - n_cut]
    load_u = load[n_cut:nt - n_cut]
    pos_d = pos[nt + n_cut:-n_cut]
    load_d = load[nt + n_cut:-n_cut]

    eps = 1e-15
    firstder_u = np.diff(load_u) / ( np.diff(pos_u) + eps )
    firstder_d = np.diff(load_d) / ( np.diff(pos_d) + eps )

    secondder_u = np.diff(np.diff(load_u)) / ( np.diff(np.diff(pos_u)) + eps )
    secondder_d = np.diff(np.diff(load_d)) / ( np.diff(np.diff(pos_d)) + eps )

    dns_n_inflc = len(np.where(np.diff(np.diff(np.where(secondder_d[:-1] * secondder_d[1:] < 0)[0])) != 0)[0])
    ups_n_inflc = len(np.where(np.diff(np.diff(np.where(secondder_u[:-1] * secondder_u[1:] < 0)[0])) != 0)[0])

    acc_vel_stat = {'st_delvel_mean': st_delvel_mean, 'st_delvel_var': st_delvel_var,
                    'st_vel_mean': st_vel_mean, 'st_vel_var': st_vel_var, 'st_acc_mean': st_acc_mean,
                    'st_acc_var': st_acc_var,
                    'ups_vel_mean': ups_vel_mean, 'ups_vel_var': ups_vel_var, 'ups_acc_mean': ups_acc_mean,
                    'ups_acc_var': ups_acc_var,
                    'dns_vel_mean': dns_vel_mean, 'dns_vel_var': dns_vel_var, 'dns_acc_mean': dns_acc_mean,
                    'dns_acc_var': dns_acc_var,
                    'ups_n_inflc': ups_n_inflc, 'dns_n_inflc': dns_n_inflc}

    return acc_vel_stat


def perim(x, y):
    # Calculates perimeter of a closed polygon
    dx2 = (np.append(x[1:], x[0]) - x) ** 2
    dy2 = (np.append(y[1:], y[0]) - y) ** 2
    return np.sum((dx2 + dy2) ** 0.5)


def periphery_area(x, y):
    # Calculates peripheral area above and below the card
    ind1 = np.where(x == x.max())
    m = ind1[0][0]

    area_above = x.max() * y.max() - 0.5 * np.sum((x[1:m] - x[:m - 1]) * (y[:m - 1] + y[1:m]))
    area_below = 0.5 * np.sum((x[m:-1] - x[m + 1:]) * (y[m:-1] + y[m + 1:]))

    return area_above, area_below


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def enumerate_cards(card_data):
    NROW, NCOl = card_data.shape


    eps = 1e-15
    
    vec_size = np.full((NROW,1),np.nan)

    surfLoadMax = np.full((NROW,1),np.nan)
    surfLoadMin = np.full((NROW,1),np.nan)

    minload = np.full((NROW,1),np.nan)
    load_atMaxPos = np.full((NROW,1),np.nan)
    maxload = np.full((NROW,1),np.nan)
    cumload = np.full((NROW,1),np.nan)

    maxpos_atMinLoad = np.full((NROW,1),np.nan)
    minpos_atMaxLoad = np.full((NROW,1),np.nan)
    maxpos = np.full((NROW,1),np.nan)
    minpos = np.full((NROW,1),np.nan)
    deviation_midFillage = np.full((NROW, 1), np.nan)
    pos_ratio = np.full((NROW,1),np.nan)

    ctr_load = np.full((NROW,1),np.nan)
    ctr_pos = np.full((NROW,1),np.nan)
    perimeter = np.full((NROW,1),np.nan)
    area_above = np.full((NROW,1),np.nan)
    area_below = np.full((NROW,1),np.nan)

    NT = 200  # len(dhCard[0])
    load_all = np.full((NROW, NT),np.nan)
    pos_all  = np.full((NROW, NT),np.nan)

    load_nall = np.full((NROW, NT),np.nan)
    pos_nall  = np.full((NROW, NT),np.nan)

    acc_vel_stat_list = ['st_delvel_mean', 'st_delvel_var',
                         'st_vel_mean', 'st_vel_var', 'st_acc_mean', 'st_acc_var',
                         'ups_vel_mean', 'ups_vel_var', 'ups_acc_mean', 'ups_acc_var',
                         'dns_vel_mean', 'dns_vel_var', 'dns_acc_mean', 'dns_acc_var',
                         'ups_n_inflc', 'dns_n_inflc']

    acc_vel_stat = {}
    for key in acc_vel_stat_list:
        acc_vel_stat[key] = np.full((NROW,1),np.nan)

    for t in range(NROW):
        try:
            dhCard = eval(card_data['DownholeCardB'].iloc[t])
            surfCard = eval(card_data['SurfaceCardB'].iloc[t])
        except:
            continue

        if len(dhCard[0]) != NT:
            continue
            print('skipped NT not 200')

        # load = list(reversed([float(i) for i in dhCard[0]]))
        # pos = list(reversed([float(i) for i in dhCard[1]]))

        load = np.asarray([float(i) for i in dhCard[0]])
        # load = load - load.min()

        pos = np.asarray([float(i) for i in dhCard[1]])
        # pos = pos - pos.min()

        #        load = [float(i) for i in POCdhCard[0]]
        #        pos = [float(i) for i in POCdhCard[1]]
        surf_load = np.asarray([float(i) for i in surfCard[0]])
        surfLoadMax[t] = surf_load.max()
        surfLoadMin[t] = surf_load.min()

        vec_size[t] =  len(pos)
        #         load_n = np.array( [(i - min(load))/(max(load)-min(load)) for i in load] )
        #         pos_n =  np.array( [(i - min(pos)) / (max(pos) - min(pos)) for i in pos] )
        load_n = (load - load.min()) / (load.max() - load.min()+eps)
        pos_n = (pos - pos.min()) / (pos.max() - pos.min() + eps)

        ctr_load[t] = load_n.mean()
        ctr_pos[t] = pos_n.mean()
        perimeter[t] = perim(pos - pos.min(), load - load.min())
        aa, ab = periphery_area(pos - pos.min(), load - load.min())
        area_above[t] = aa.mean()
        area_below[t] = ab.mean()

        minload[t] = load.min()
        maxload[t] = load.max()

        cumload[t] = load.sum()

        ind0 = np.where(load_n <= 0.05)
        pos_load0 = pos[ind0[0]]
        maxpos_atMinLoad[t] = max(pos_load0)
        fillage = max(pos_n[ind0[0]])
        deviation_midFillage[t] = abs(0.5 - fillage)

        ind0 = np.where(load_n >= 0.95)
        pos_load1 = pos[ind0[0]]
        minpos_atMaxLoad[t] = min(pos_load1)

        ind0 = np.where(pos_n >= 0.95)
        load_pos0 = load[ind0[0]]
        load_atMaxPos[t] = min(load_pos0)


        ind0 = np.where(load_n >= 0.95)
        maxpos_load1 = max(pos[ind0[0]])
        pos_ratio[t] = max(pos_load0) / maxpos_load1

        maxpos[t] = pos.max()
        minpos[t] = pos.min()

        load_all[t,:] = load
        pos_all[t,:] = pos

        load_nall[t,:] = load_n
        pos_nall[t,:] = pos_n

        # Derivatives
        av_stat = calc_card_derivatives(pos_n, load_n)

        for key in acc_vel_stat:
            acc_vel_stat[key][t] = av_stat[key]

    output = {'load': load_all,
              'load_norm': load_nall,
              'position': pos_all,
              'position_norm': pos_nall,
              'load_atMaxPos': load_atMaxPos,
              'surfLoadMax': surfLoadMax,
              'surfLoadMin': surfLoadMin,
              'position_min': minpos,
              'position_max': maxpos,
              'maxPosition_at_minLoad': maxpos_atMinLoad,
              'minPosition_at_maxLoad': minpos_atMaxLoad,
              'deviation_midFillage':deviation_midFillage,
              'load_min': minload,
              'load_max': maxload,
              'load_sum': cumload,
              'load_norm_center': ctr_load,
              'position_norm_center': ctr_pos,
              'perimeter': perimeter,
              'area_above': area_above,
              'area_below': area_below,
              'maxpos_ratio': pos_ratio,
              'cumsum_load_norm': np.cumsum(load_nall, axis=1)[:, 0::10],
              'cumsum_position_norm': np.cumsum(pos_nall, axis=1)[:, 0::10]
              }

    output['area'] = [PolyArea(pos_all[i, :], load_all[i, :]) for i in range(load_all.shape[0])]
    output['areaToPerimeter'] = np.array(output['area']) / np.array(output['perimeter'])

    output.update(acc_vel_stat)
    # for key in acc_vel_stat:
    #     output[key] = acc_vel_stat[key]

    return output
	
def parallelize(data):    
    cores = cpu_count() #Number of CPU cores on your system
    partitions = cores #Define as many partitions as you want
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data=[] 
    data.append(pool.map(enumerate_cards, data_split))
    pool.close()
    pool.join()
    data_out = data[0][0]
    for i in range(1,len(data[0])):
        for key in data_out:
            if data_out[key].shape[0]>0 and data[0][i][key].shape[0]>0:
                data_out[key] = np.vstack((data_out[key], data[0][i][key]))
            elif data_out[key].shape[0]==0 and data[0][i][key].shape[0]>0:
                data_out[key] = data[0][i][key]

    return data_out


def enumerate_cards_prll(card_data,prll):
    if prll:
        card_enum = parallelize(card_data)
    else:
        if type(card_data)==dict:
            for key in card_data:
                card_enum[key] = enumerate_cards(card_data[key])
        else:
            card_enum = enumerate_cards(card_data)


    return card_enum
# with open('well_card_test.pkl','rb') as f:
#     wcd = pickle.load(f)
# card_enum_4clstring = enumerate_cards(wcd)
#
# card_enum_4clstring.keys()
