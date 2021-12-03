# Функции, разработанные в рамках проекта P05, модуля P05_03_MVP_Baseline
#
#

import numpy as np
import pandas as pd
import time
import category_encoders as ce
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
RANDOM_SEED = 42


def initial_transformation(data):

    rename_dict= {'bodyType':'body', 'car_url': 'url', 'engineDisplacement':'displacement', 'enginePower':'power',  'fuelType':'fuel', 'modelDate':'model_year', 'model_name':'model', 'numberOfDoors': 'doors', 'priceCurrency':'curr', 'productionDate':'manuf_year', 'vehicleConfiguration' : 'configuration', 'vehicleTransmission': 'transmission', 'Владельцы':'num_owners', 'Владение':'ownership', 'ПТС':'car_license', 'Привод':'gear', 'Руль':'steering', 'Состояние':'condition', 'Таможня':'duties'}
    data.rename( columns= rename_dict, inplace=True)

    # print(data.columns)
    cols2drop = ['image', 'parsing_unixtime', 'description', 'name', 'complectation_dict', 'equipment_dict', 'model_info', 'curr', 'super_gen', 'configuration', 'condition', 'duties', 'ownership']
    data.drop(cols2drop, axis=1, inplace=True )
    if 'location' in data.columns:
        data.drop(['location'], axis=1, inplace=True)

    data.body = data.body.apply(lambda x: x.lower().split()[0].strip() if isinstance(x, str) else x)
    bt = 'bodytype'; btl = list(data.body.unique())
    if bt in btl :
        data.drop(data.loc[data.body==bt].index, inplace=True)
    # print(list(data.body.unique()))
    body_dict = {'кабриолет': 'coupe', 'седан': 'sedan', 'внедорожник': 'SUV', 'купе': 'coupe', 'родстер': 'coupe',
                 'хэтчбек': 'f_back', 'лифтбек': 'f_back', 'универсал': 'wagon', 'пикап': 'SUV', 'минивэн': 'MPV',
                 'компактвэн': 'MPV', 'купе-хардтоп': 'coupe', 'фургон': 'MPV', 'микровэн': 'MPV', 'тарга': 'coupe', 'фастбек': 'f_back', 'лимузин': 'sedan', 'седан-хардтоп': 'sedan'}
    data['body_type'] = data['body'].apply(lambda x: body_dict[x])

    data.brand = data.brand.apply(lambda x: x.upper())
    data.brand = data.brand.apply(lambda x: 'MERCEDES-BENZ' if x == 'MERCEDES' else x)
    color_dict = {'040001': 'чёрный', 'FAFBFB': 'белый', '97948F': 'серый', 'CACECB': 'серебристый', '0000CC': 'синий',
                  '200204': 'коричневый', 'EE1D19': 'красный', '007F00': 'зелёный', 'C49648': 'бежевый',
                  '22A0F8': 'голубой', '660099': 'пурпурный', 'DEA522': 'золотистый', '4A2197': 'фиолетовый',
                  'FFD600': 'жёлтый', 'FF8649': 'оранжевый', 'FFC0CB': 'розовый'}
    data.color.replace(to_replace=color_dict, inplace=True)

    data.displacement = data.displacement.apply(lambda x: x.replace(" LTR", "0.0 LTR") if x == " LTR" else x)
    data.displacement = data.displacement.apply(lambda x: float(x.replace("LTR", "")) if isinstance(x, str) else x)
    data.power = data.power.apply(lambda x: float(x.replace("N12", "")) if isinstance(x, str) else x)

    fuel_dict={'DIESEL':'D', 'дизель':'D', 'GASOLINE':'B', 'бензин':'B', 'HYBRID':'H', 'гибрид':'H', 'ELECTRO':'E', 'электро':'E', 'LPG':'G', 'газ':'G'}
    data.fuel = data.fuel.apply(lambda x: fuel_dict[x])

    data.mileage = data.mileage.apply(lambda x: float(x))
    data.mileage = data.mileage.apply(lambda x: 10.0 if x < 10.0 else x)

    data.model_year = data.model_year.apply(lambda x: int(x))
    data.manuf_year = data.manuf_year.apply(lambda x: int(x))
    if 'price' in data.columns:
        data.price = data.price.apply(lambda x: int(x))

    data.model = data.model.apply(lambda x: x.upper())

    data.doors = data.doors.apply(lambda x: int(x))
    data.doors = data.doors.apply( lambda x: 2 if x ==0 else x)

    data.num_owners.fillna(0, inplace=True)
    data.num_owners = data.num_owners.apply( lambda x: int(x[0]) if isinstance(x, str) else int(x))
    data.num_owners = data.num_owners.apply(lambda x: 3 if x>2 else x)

    transmission_dict = {'AUTOMATIC':'AT', 'автоматическая':'AT', 'ROBOT': 'AMT', 'роботизированная':'AMT', 'MECHANICAL':'MT', 'механическая':'MT', 'VARIATOR':'CVT', 'вариатор':'CVT'}
    data.transmission = data.transmission.apply(lambda x: transmission_dict[x])

    data.car_license.fillna('ORIGINAL', inplace=True)
    license_dict={'ORIGINAL':'ORG', 'Оригинал':'ORG', 'DUPLICATE':'DPL', 'Дубликат':'DPL'}
    data.car_license = data.car_license.apply(lambda x: license_dict[x])
    data['orig_license'] = data.car_license.apply(lambda t: 1 if t == 'ORG' else 0)

    gear_dict = {'REAR_DRIVE':'RWD', 'ALL_WHEEL_DRIVE':'AWD', 'FORWARD_CONTROL': 'FWD', 'задний':'RWD', 'полный':'AWD', 'передний': 'FWD'}
    data.gear = data.gear.apply(lambda x: gear_dict[x])

    steering_dict = {'LEFT':'L', 'RIGHT':'R', 'Левый':'L', 'Правый':'R'}
    data.steering = data.steering.apply(lambda x: steering_dict[x])
    data['left_steering'] =  data.steering.apply(lambda t: 1 if t =='L' else 0)

    data.vendor = data.vendor.apply(lambda x: 'EUROPEAN' if x == 'AMERICAN' else x)
    data.drop(['car_license', 'steering', 'vendor'], axis=1, inplace=True)

    return data


#
def parse_simple_encoder(data, cols):
    parse_dict = dict()
    for col in cols:
        di = dict()
        li = list(data[col].unique())
        n = 0
        for v in li:
            di[v]=n
            n+=1
        parse_dict[col] = di
    return parse_dict


def apply_encoder(data, cols, enc_dic):
    li = []
    for c in cols:
        enc = enc_dic[c]
        vals = enc.keys()
        # print(enc)
        nc = c+'_enc'
        data[nc] = data[c].apply(lambda x: enc[x] if x in vals else 0)
        li.append(nc)
    return li, data


def parse_ranging_encoder(data, cols):
    parse_dict = dict()
    for c in cols:
        di = dict()
        li = list(data[c].value_counts(ascending=True).index)
        n = 0
        for v in li:
            di[v]=n
            n+=1
        parse_dict[c] = di
    return parse_dict


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def eval_data(Xdf,ydf,name):
    was = time.perf_counter()
    X = Xdf.to_numpy()
    y = ydf.to_numpy()
    models = {
        # models with default settings
        # criterion = 'absolute_error' - excluded
        'ExtEns' : ExtraTreesRegressor(n_estimators=100,  n_jobs=-1, random_state = RANDOM_SEED),
        # 'AdaBst'  : AdaBoostRegressor(n_estimators=500, random_state=RANDOM_SEED),
        'GradBst' : GradientBoostingRegressor(n_estimators=1000,  random_state=RANDOM_SEED),
        'RanFrst' : RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED )
    }
    lm = models.keys()
    metrics = pd.Series(data=np.zeros(len(lm)), index=lm, name=name, dtype='float64')
    N = 5
    print('Eval:', end=' ')
    for mname in lm:
        print(mname, end=' ')
        #,
        kf = KFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED)
        for trn_index, tt_index in kf.split(X):
            X_trn = X[trn_index] ; X_tt =  X[tt_index]
            y_trn = y[trn_index] ; y_tt =  y[tt_index]
            # rfr_log.fit(X_train, np.log(y_train))
            # predict_rf_log = np.exp(rfr_log.predict(X_test))
            models[mname].fit(X_trn, np.log(y_trn))
            y_prd = np.exp(models[mname].predict(X_tt))
            metrics[mname] += mape(y_tt, y_prd) * 100.0
        metrics[mname]  /= N
    print('done. {:.4f} sec'.format(time.perf_counter() - was))
    return metrics

def eval_model(Xdf,ydf,model):
    was = time.perf_counter()
    X = Xdf.to_numpy()
    y = ydf.to_numpy()
    metric = 0
    N = 5
    # print('Eval:', end=' ')
    kf = KFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED)
    for trn_index, tt_index in kf.split(X):
        X_trn = X[trn_index] ; X_tt =  X[tt_index]
        y_trn = y[trn_index] ; y_tt =  y[tt_index]
        model.fit(X_trn, np.log(y_trn))
        y_prd = np.exp(model.predict(X_tt))
        metric += mape(y_tt, y_prd) * 100.0
    metric /= N
    print('done. {:.4f} sec'.format(time.perf_counter() - was))
    return metric

# По каким то причинам внешний источник данных не захотел однозначно индексироваться в виде датафрейма
# Простой map датафрейма не сработал из-за ошибки: Reindexing only valid with uniquely valued Index objects
# Пришлось выходить из положения
def make_feature_dict(ref,feature):
    a_dict = dict()
    for i in ref.index:
        a_dict[ref.model[i]] =  ref[feature][i]
    return a_dict


def add_external_features(data,ref):
    for f in ['segment', 'geo', 'eur_size']:
        data[f] = data.model.map( make_feature_dict(ref,f) )
    # Понимаю, что спортивные машинки тоже отличаются размером, но принимаю решение что в среднем это компактные машинки.
    size_dict = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'S':3 }
    data['size_cat'] = data.eur_size.map(size_dict)
    data['sport_car'] = data.eur_size.apply(lambda s: 1 if s =='S' else 0)
    data['euro_car'] = data.geo.apply(lambda x: 1 if x =='EUR' else 0)
    data['premium'] =  data.segment.apply(lambda t: 1 if t =='Prem' else 0)
    data.drop(['segment', 'geo', 'eur_size'], axis=1, inplace = True)
    return data


# показ квантилей для выбранного признака и расчет возможных ограничений для выбросов
def show_quantile_hurdle(ser):
    qv = ser.quantile(q=[0.0,0.25,0.5,0.75,0.95,1.0])
    print(qv)
    q1 = qv[0.25]; q3 = qv[0.75]; iqr = q3-q1;
    print('\n','Low hurdle: {:.4f}'.format(q1-1.5*iqr),'high hurdle: {:.4f}'.format(q3+1.5*iqr))


# добавление новых признаков
def engineer_features(data):
    # Возраст
    data['age'] = 2020 + data.train - data.manuf_year
    data.age = data.age.apply(lambda x: x if x <=30. else 30.0 )
    # Пробег
    data['ann_mil'] = data.apply(lambda t: 5 if t['age']<1 else t['mileage']/t['age'], axis=1)
    data.ann_mil = data.ann_mil.apply(lambda x: x if x < 33800 else 33800)
    data['intensity'] = data.ann_mil.apply(lambda x: 0 if x <= 15000 else 1 )
    data.mileage = data.mileage.apply(lambda m: m if m <420000 else 420000)
    # Мощность и удельная мощность
    data['power_size'] = data.power / data.size_cat
    data['vol_power'] = data.apply(lambda t: 0 if t['displacement']<0.1 else t['power']/t['displacement'], axis=1)
    # data.vol_power = data.vol_power.apply(lambda x: x if x < 2.156 else 2.16)
    return data


#Целевое кодирование по выбранной модели кодировщика
def target_encode(data,col,enc):
    tgt_name = col +'_tgt'
    x_col = dict()
    data[tgt_name] = 0
    column = data.query('train==1')[col].to_numpy()
    target = np.log(data.query('train==1').price.to_numpy())
    i = 0
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    for trn_index, _ in kf.split(column):
        enc.fit( column[trn_index], target[trn_index] )
        x_col[i] = enc.transform(data[col].to_numpy())
        i += 1
    data[tgt_name] = (x_col[0] + x_col[1] + x_col[2])/3
    return tgt_name, data


# Кодирование оставщихся категориальных переменных
def encode_low_card_categories(data):
    low_card_cat = ['fuel',  'transmission',  'gear', 'body_type']
    dummies = pd.get_dummies(data[low_card_cat])
    dum_cat = list(dummies.columns)
    data = pd.concat([data, dummies], axis=1)
    return dum_cat, data

def encode_high_card_catrgories(data):
    high_card_cat = ['brand', 'color', 'model']
    enc = ce.GLMMEncoder(random_state=RANDOM_SEED)
    col_list = []
    for cat in high_card_cat:
        nn, data = target_encode(data, cat, enc)
        col_list.append(nn)
    return  col_list, data
