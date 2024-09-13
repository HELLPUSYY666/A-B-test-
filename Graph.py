import pandas as pd  # для чтения csv
from scipy import stats  # вычисление доверительных интервалов
import numpy as np  # для массивов
import matplotlib.pyplot as plt  # построение графиков
import seaborn as sns  # сложные графики

dtypes_money = {
    'user_id': 'object',
    'date': 'object',
    'money': 'float64',
}
dtypes_cheaters = {
    'user_id': 'object',
    'cheaters': 'int64',
}
dtypes_platforms = {
    'user_id': 'object',
    'platform': 'object',
}
dtypes_cash = {
    'user_id': 'object',
    'date': 'object',
    'cash': 'int64',
}
dtypes_abgroup = {
    'user_id': 'object',
    'group': 'object',
}
money_df = pd.read_csv('/Users/zakariyapolevchishikov/PycharmProjects/For Sysya/tablici /Money.csv', dtype=dtypes_money,
                       )
cheaters_df = pd.read_csv('/Users/zakariyapolevchishikov/PycharmProjects/For Sysya/tablici /Cheaters.csv',
                          dtype=dtypes_cheaters,
                          )
platforms_df = pd.read_csv('/Users/zakariyapolevchishikov/PycharmProjects/For Sysya/tablici /Platforms.csv',
                           dtype=dtypes_platforms,
                           )
cash_df = pd.read_csv('/Users/zakariyapolevchishikov/PycharmProjects/For Sysya/tablici /Cash.csv', dtype=dtypes_cash,
                      )
abgroup_df = pd.read_csv('/Users/zakariyapolevchishikov/PycharmProjects/For Sysya/tablici /ABgroup.csv',
                         dtype=dtypes_abgroup, )

avg_spending = cash_df.groupby('user_id')['cash'].mean()
threshold = avg_spending.mean() * 5  # порог в 5 раз выше среднего
potential_cheaters = avg_spending[avg_spending > threshold].index  # идентификация потенциальных читеров
# print(potential_cheaters)
# print(cheaters_df)
clean_money = money_df[~money_df['user_id'].isin(potential_cheaters)]
clean_money_x2 = clean_money[~clean_money['user_id'].isin(cheaters_df)]
# print(clean_money)
unite_plat_abgroup = pd.merge(platforms_df, abgroup_df, on='user_id', how='left').drop_duplicates()

test_group_df = unite_plat_abgroup[unite_plat_abgroup['group'] == 'test']
control_group_df = unite_plat_abgroup[unite_plat_abgroup['group'] == 'control']

unite_test = pd.merge(test_group_df, clean_money_x2, on='user_id', how='left').drop_duplicates()
unite_control = pd.merge(control_group_df, clean_money_x2, on='user_id', how='left').drop_duplicates()


def calculate_arpu_arppu(df):
    total_revenue = df.groupby('user_id')['money'].sum()
    paying_users = total_revenue[total_revenue > 0]
    arpu = total_revenue.mean()
    arppu = paying_users.mean() if not paying_users.empty else 0
    return arpu, arppu


def calculate_average_spending(df):
    return df['money'].mean()


arpu_test, arppu_test = calculate_arpu_arppu(unite_test)
arpu_control, arppu_control = calculate_arpu_arppu(unite_control)

avg_spending_test = calculate_average_spending(unite_test)
avg_spending_control = calculate_average_spending(unite_control)

print(f"Test Group ARPU: {arpu_test}")
print(f"Test Group ARPPU: {arppu_test}")
print(f"Test Group Avg Spending: {avg_spending_test}")

print(f"Control Group ARPU: {arpu_control}")
print(f"Control Group ARPPU: {arppu_control}")
print(f"Control Group Avg Spending: {avg_spending_control}")


def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = stats.sem(data)
    interval = std_err * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - interval, mean + interval


arpu_test_ci = confidence_interval(unite_test['money'])
arpu_control_ci = confidence_interval(unite_control['money'])

platform_analysis_test = unite_test.groupby('platform').agg({'money': ['mean', 'sum']})
platform_analysis_control = unite_control.groupby('platform').agg({'money': ['mean', 'sum']})

print("Test Group Platform Analysis:")
print(platform_analysis_test)

print("Control Group Platform Analysis:")
print(platform_analysis_control)

# Гистограмма распределения ARPU для тестовой и контрольной групп
plt.figure(figsize=(12, 8))
sns.histplot(unite_test['money'], kde=True, color='blue', label='Test Group', stat='density', alpha=0.6)
sns.histplot(unite_control['money'], kde=True, color='orange', label='Control Group', stat='density', alpha=0.6)
plt.axvline(arpu_test, color='blue', linestyle='dashed', linewidth=2, label=f'Test Group ARPU: {arpu_test:.2f}')
plt.axvline(arpu_control, color='orange', linestyle='dashed', linewidth=2, label=f'Control Group ARPU: {arpu_control:.2f}')
plt.xlabel('ARPU')
plt.ylabel('Density')
plt.title('Distribution of ARPU by Group')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.boxplot(data=unite_test[['money']].rename(columns={'money': 'ARPU'}), color='blue')
sns.boxplot(data=unite_control[['money']].rename(columns={'money': 'ARPU'}), color='orange')
plt.title('Boxplot of ARPU')

plt.subplot(1, 2, 2)
sns.boxplot(data=unite_test[['money']].rename(columns={'money': 'ARPPU'}), color='blue')
sns.boxplot(data=unite_control[['money']].rename(columns={'money': 'ARPPU'}), color='orange')
plt.title('Boxplot of ARPPU')

plt.show()


plt.figure(figsize=(12, 8))
platform_analysis_test['money']['sum'].plot(kind='bar', color='blue', alpha=0.7, label='Test Group')
platform_analysis_control['money']['sum'].plot(kind='bar', color='orange', alpha=0.7, label='Control Group')
plt.xlabel('Platform')
plt.ylabel('Total Spending')
plt.title('Total Spending by Platform')
plt.legend()
plt.grid(True)
plt.show()

