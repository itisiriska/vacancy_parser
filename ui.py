import numpy as np
import pandas as pd

pd.set_option('max_colwidth', 400)

import panel as pn

from bokeh.models.widgets.tables import NumberFormatter
from solution import VacancySplitter

pn.extension()


def get_data(event):
    df = pd.read_excel('Датасет.xlsx')
    total_responsibilities.value = len(df)
    total_empty_requirements.value = df['requirements(Требования к соискателю)'].isna().sum()
    total_empty_terms.value = df['terms(Условия)'].isna().sum()

    df_bundle_info.value = df[
        ['responsibilities(Должностные обязанности)', 'requirements(Требования к соискателю)', 'terms(Условия)']]

    upload_button.disabled = True
    check_values_button.button_type = 'default'
    check_values_button.disabled = False


def get_missing_values(event):
    data = pd.read_excel('Датасет.xlsx')
    data.index = data['id']
    test = pd.read_pickle('test.pkl')
    missing_vals = data.loc[test['id'].unique()][['responsibilities(Должностные обязанности)']]

    missing_vals[['requirements(Требования к соискателю)', 'terms(Условия)']] = np.nan
    df_bundle_info.value = missing_vals

    check_values_button.disabled = True
    run_model_button.button_type = 'default'
    run_model_button.disabled = False


def run_model(event):
    data = pd.read_excel('Датасет.xlsx')
    data.index = data['id']
    vs = VacancySplitter()
    df_train = pd.read_pickle("train.pkl")
    vs.train(df_train['responsibilities_bigrams'], df_train['class'])
    test = pd.read_pickle('test.pkl')
    predict = vs.predict(test)
    predict['responsibilities'] = data.loc[predict.index]['responsibilities(Должностные обязанности)']
    df_bundle_info.value = predict[['responsibilities', 'reqiurements', 'terms']]
    run_model_button.disabled = True
    upload_button.button_type = 'default'
    upload_button.disabled = False


bokeh_formatters = {
    'churn_decrease': NumberFormatter(format='0.00')
}

pn.extension(sizing_mode="stretch_height")

upload_button = pn.widgets.Button(name='Upload Data', button_type='primary', height=400)
check_values_button = pn.widgets.Button(name='Find missing values', button_type='default', disabled=True, height=400)
run_model_button = pn.widgets.Button(name='Run Model', button_type='default', disabled=True, height=400)

start_info = pn.Column(upload_button, check_values_button, run_model_button, align='center', width=150, height=150,
                       visible=True, css_classes=['panel-widget-box'])

total_responsibilities = pn.indicators.Number(name='Total responsibilitie:', title_size='14pt', font_size='12pt',
                                              sizing_mode="stretch_both")
total_empty_requirements = pn.indicators.Number(name='Total empty requirements:', title_size='14pt', font_size='12pt',
                                                sizing_mode="stretch_both")
total_empty_terms = pn.indicators.Number(name='Total empty terms:', title_size='14pt', font_size='12pt',
                                         sizing_mode="stretch_both")

total = pn.Row(total_responsibilities, total_empty_requirements, total_empty_terms, align='center', visible=True,
               css_classes=['panel-widget-box'])

start_info = pn.Row(start_info, total)
text_base_info = pn.widgets.StaticText(name='Data Results', value='', style={'font-size': 'large'})

df_bundle_info = pn.widgets.Tabulator(name='', height=400, width=400, theme='site', layout='fit_columns',
                                      show_index=False, disabled=False, sizing_mode="stretch_width",
                                      formatters=bokeh_formatters)

analysis_results = pn.Column(start_info, text_base_info, df_bundle_info, visible=True, sizing_mode="stretch_both")

upload_button.on_click(get_data)
check_values_button.on_click(get_missing_values)
run_model_button.on_click(run_model)

# pn.Row(file_input)
template = pn.template.FastListTemplate(title='Vacancy Split', main=[analysis_results],
                                        header_background='#009624', background_color='#e8f5e9',
                                        neutral_color='#009688', accent_base_color='#009688'
                                        ).show()
