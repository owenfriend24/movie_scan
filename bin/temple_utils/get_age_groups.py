import pandas as pd

def get_children():
    children =  [34, 41, 53, 64, 65, 66, 68, 69, 70, 78,
            84, 92, 94, 95, 97, 98, 107, 113, 114, 115, 116,
                 122, 125, 127, 129, 130, 132, 133, 135, 136]
    return[f"temple{str(sub).zfill(3)}" for sub in children]

def get_adolescents():
    adolescents = [29, 30, 32, 33, 35, 36, 38, 42, 45, 51,
            60, 63, 79, 82, 83, 85, 90, 91, 93, 96,
            103, 109, 110, 111, 112, 117, 121, 126, 128, 131]
    return [f"temple{str(sub).zfill(3)}" for sub in adolescents]

def get_adults():
    adults = [16, 19, 20, 22, 23, 24, 25, 37, 50, 56,
            57, 58, 59, 71, 72, 73, 74, 75, 76, 87,
            88, 89, 99, 105, 106, 108, 119, 120, 123, 124]
    return [f"temple{str(sub).zfill(3)}" for sub in adults]

def get_all_subjects():
    return get_children() + get_adolescents() + get_adults()

def get_age_years(subject):
    ref = pd.read_csv('/home1/09123/ofriend/analysis/temple/bin/templates/randomise_measures.csv')
    sub_ref = ref[ref['subject'] == subject]
    age = sub_ref.age.values[0]
    return age


