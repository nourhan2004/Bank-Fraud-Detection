"""
================================================
  نظام كشف الاحتيال البنكي - مشروع التخرج
  المراحل 2 و 3 و 4: تحليل البيانات + التجهيز + التدريب
================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  المرحلة 2: تحليل البيانات واستكشافها (EDA)
# ============================================================
print("=" * 55)
print("  المرحلة 2: تحليل البيانات (EDA)")
print("=" * 55)

df = pd.read_csv('credit_card_fraud_10k.csv')

print(f"\n📋 شكل البيانات: {df.shape[0]} صف × {df.shape[1]} عمود")
print(f"\n📌 أسماء الأعمدة:\n{df.columns.tolist()}")
print(f"\n📊 أنواع البيانات:\n{df.dtypes}")
print(f"\n🔍 أول 5 صفوف:\n{df.head()}")
print(f"\n📈 إحصائيات عامة:\n{df.describe().round(2)}")

# التحقق من القيم المفقودة
missing = df.isnull().sum()
print(f"\n❓ القيم المفقودة:\n{missing}")
if missing.sum() == 0:
    print("✅ لا توجد قيم مفقودة — البيانات نظيفة!")

# توزيع الاحتيال
fraud_counts = df['is_fraud'].value_counts()
fraud_pct = df['is_fraud'].mean() * 100
print(f"\n🚨 توزيع الاحتيال:")
print(f"   معاملات طبيعية  : {fraud_counts[0]:,} ({100-fraud_pct:.1f}%)")
print(f"   معاملات احتيالية: {fraud_counts[1]:,} ({fraud_pct:.1f}%)")
print(f"   ⚠️  البيانات غير متوازنة — ستُعالَج في المرحلة 3")

# ---- الرسوم البيانية ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('تحليل بيانات المعاملات البنكية', fontsize=16, fontweight='bold', y=1.01)

colors = ['#2196F3', '#F44336']
labels = ['Legitimate (طبيعية)', 'Fraudulent (احتيالية)']

# 1. توزيع الاحتيال
ax = axes[0, 0]
bars = ax.bar(labels, fraud_counts.values, color=colors, edgecolor='white', width=0.5)
ax.set_title('توزيع المعاملات', fontweight='bold')
ax.set_ylabel('العدد')
for bar, val in zip(bars, fraud_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}', ha='center', fontweight='bold')
ax.set_ylim(0, max(fraud_counts.values) * 1.15)

# 2. توزيع قيمة المعاملة حسب النوع
ax = axes[0, 1]
for val, color, label in zip([0, 1], colors, labels):
    subset = df[df['is_fraud'] == val]['amount']
    ax.hist(subset, bins=40, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_title('توزيع قيمة المعاملة', fontweight='bold')
ax.set_xlabel('القيمة ($)')
ax.set_ylabel('التكرار')
ax.legend(fontsize=8)

# 3. المعاملات حسب الساعة
ax = axes[0, 2]
fraud_by_hour = df.groupby(['transaction_hour', 'is_fraud']).size().unstack(fill_value=0)
fraud_by_hour.columns = ['طبيعية', 'احتيالية']
ax.plot(fraud_by_hour.index, fraud_by_hour['طبيعية'], color=colors[0],
        linewidth=2, label='طبيعية', marker='o', markersize=3)
ax2_twin = ax.twinx()
ax2_twin.plot(fraud_by_hour.index, fraud_by_hour['احتيالية'], color=colors[1],
              linewidth=2, label='احتيالية', marker='s', markersize=3)
ax.set_title('المعاملات حسب الساعة', fontweight='bold')
ax.set_xlabel('الساعة')
ax.set_ylabel('عدد المعاملات الطبيعية', color=colors[0])
ax2_twin.set_ylabel('عدد المعاملات الاحتيالية', color=colors[1])

# 4. معدل الاحتيال حسب نوع التاجر
ax = axes[1, 0]
fraud_by_cat = df.groupby('merchant_category')['is_fraud'].agg(['sum','count'])
fraud_by_cat['rate'] = (fraud_by_cat['sum'] / fraud_by_cat['count'] * 100).round(2)
fraud_by_cat = fraud_by_cat.sort_values('rate', ascending=True)
bars = ax.barh(fraud_by_cat.index, fraud_by_cat['rate'],
               color=['#FF7043', '#FF8A65', '#FFA726', '#FFCC02', '#66BB6A'])
ax.set_title('معدل الاحتيال حسب نوع التاجر (%)', fontweight='bold')
ax.set_xlabel('معدل الاحتيال (%)')
for bar, val in zip(bars, fraud_by_cat['rate']):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val}%', va='center', fontsize=9)

# 5. device_trust_score مقابل الاحتيال
ax = axes[1, 1]
df.boxplot(column='device_trust_score', by='is_fraud', ax=ax,
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
ax.set_title('درجة الثقة بالجهاز مقابل الاحتيال', fontweight='bold')
ax.set_xlabel('0=طبيعية   1=احتيالية')
ax.set_ylabel('device_trust_score')
plt.sca(ax)
plt.title('درجة الثقة بالجهاز مقابل الاحتيال')

# 6. Correlation Heatmap
ax = axes[1, 2]
num_cols = ['amount', 'transaction_hour', 'foreign_transaction',
            'location_mismatch', 'device_trust_score', 'velocity_last_24h',
            'cardholder_age', 'is_fraud']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax,
            cmap='RdYlGn', center=0, linewidths=0.5,
            annot_kws={'size': 7})
ax.set_title('مصفوفة الارتباط', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ تم حفظ الرسوم البيانية في: eda_analysis.png")


# ============================================================
#  المرحلة 3: تجهيز البيانات (Preprocessing)
# ============================================================
print("\n" + "=" * 55)
print("  المرحلة 3: تجهيز البيانات (Preprocessing)")
print("=" * 55)

# 1. ترميز العمود النصي
print("\n🔤 [1] ترميز merchant_category ...")
le = LabelEncoder()
df['merchant_category_enc'] = le.fit_transform(df['merchant_category'])
print(f"   الفئات: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 2. تحديد الميزات والهدف
FEATURES = ['amount', 'transaction_hour', 'merchant_category_enc',
            'foreign_transaction', 'location_mismatch',
            'device_trust_score', 'velocity_last_24h', 'cardholder_age']
TARGET = 'is_fraud'

X = df[FEATURES]
y = df[TARGET]
print(f"\n🎯 [2] الميزات المستخدمة ({len(FEATURES)}):\n   {FEATURES}")

# 3. معالجة عدم التوازن (Oversampling يدوي)
print(f"\n⚖️  [3] معالجة عدم التوازن ...")
print(f"   قبل المعالجة: طبيعية={y.value_counts()[0]} | احتيالية={y.value_counts()[1]}")

df_temp = pd.concat([X, y], axis=1)
majority_class = df_temp[df_temp[TARGET] == 0]
minority_class = df_temp[df_temp[TARGET] == 1]
minority_upsampled = resample(minority_class,
                              replace=True,
                              n_samples=len(majority_class),
                              random_state=42)
df_balanced = pd.concat([majority_class, minority_upsampled])
X_balanced = df_balanced[FEATURES]
y_balanced = df_balanced[TARGET]
print(f"   بعد المعالجة : طبيعية={y_balanced.value_counts()[0]} | احتيالية={y_balanced.value_counts()[1]}")
print(f"   ✅ البيانات متوازنة الآن!")

# 4. تقسيم التدريب والاختبار
print(f"\n✂️  [4] تقسيم البيانات (80% تدريب / 20% اختبار) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"   بيانات التدريب : {X_train.shape[0]:,} صف")
print(f"   بيانات الاختبار: {X_test.shape[0]:,} صف")

# 5. تطبيع الأرقام
print(f"\n📏 [5] تطبيع القيم (StandardScaler) ...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"   ✅ تم تطبيع جميع الأعمدة الرقمية")


# ============================================================
#  المرحلة 4: تدريب النماذج ومقارنتها
# ============================================================
print("\n" + "=" * 55)
print("  المرحلة 4: تدريب النماذج ومقارنتها")
print("=" * 55)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n🤖 تدريب نموذج: {name} ...")
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results[name]       = {'accuracy': acc, 'auc': auc, 'f1': f1,
                           'precision': precision, 'recall': recall,
                           'cm': cm, 'y_proba': y_proba}
    trained_models[name] = model

    print(f"   ✅ Accuracy : {acc*100:.2f}%")
    print(f"   ✅ AUC Score: {auc:.4f}")
    print(f"   ✅ F1 Score : {f1:.4f}")
    print(f"   TP={tp} | FP={fp} | TN={tn} | FN={fn}")

# ---- جدول المقارنة ----
print("\n" + "=" * 55)
print("  📊 جدول مقارنة النماذج")
print("=" * 55)
print(f"\n{'النموذج':<22} {'Accuracy':>10} {'AUC':>8} {'F1':>8} {'Precision':>11} {'Recall':>9}")
print("-" * 70)
best_name = max(results, key=lambda k: results[k]['auc'])
for name, r in results.items():
    star = " ⭐ أفضل نموذج" if name == best_name else ""
    print(f"{name:<22} {r['accuracy']*100:>9.2f}% {r['auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}{star}")

print(f"\n🏆 أفضل نموذج: {best_name}")

# ---- رسوم بيانية للمقارنة ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('مقارنة نماذج تعلم الآلة', fontsize=14, fontweight='bold')

model_names  = list(results.keys())
model_colors = ['#42A5F5', '#EF5350', '#66BB6A']

# 1. مقارنة المقاييس
ax = axes[0]
metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
metric_labels = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
x = np.arange(len(metrics))
width = 0.25
for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.85)
ax.set_title('مقارنة المقاييس', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel('القيمة')
ax.legend(fontsize=8)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)

# 2. Confusion Matrix لأفضل نموذج
ax = axes[1]
cm = results[best_name]['cm']
sns.heatmap(cm, annot=True, fmt='d', ax=ax,
            cmap='Blues', linewidths=1,
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'],
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
ax.set_ylabel('القيمة الحقيقية')
ax.set_xlabel('القيمة المتوقعة')

# 3. ROC Curve
ax = axes[2]
for name, color in zip(model_names, model_colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC={results[name]['auc']:.3f})")
ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
ax.set_title('منحنى ROC', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ تم حفظ رسوم المقارنة في: model_comparison.png")

# ---- Feature Importance لأفضل نموذج ----
best_model = trained_models[best_name]
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(9, 5))
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)
    feature_labels = ['القيمة', 'ساعة المعاملة', 'نوع التاجر',
                      'خارجية؟', 'تعارض موقع', 'ثقة الجهاز',
                      'سرعة المعاملات', 'عمر العميل']
    bars = ax.barh([feature_labels[i] for i in sorted_idx],
                   importance[sorted_idx],
                   color=plt.cm.RdYlGn(importance[sorted_idx] / importance.max()))
    ax.set_title(f'أهمية كل ميزة في نموذج {best_name}', fontweight='bold', fontsize=13)
    ax.set_xlabel('درجة الأهمية')
    for bar, val in zip(bars, importance[sorted_idx]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ تم حفظ رسم أهمية الميزات في: feature_importance.png")

# ============================================================
#  حفظ النموذج والأدوات
# ============================================================
print("\n" + "=" * 55)
print("  💾 حفظ النموذج")
print("=" * 55)

joblib.dump(trained_models[best_name], 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(FEATURES, 'feature_names.pkl')

print(f"\n✅ تم حفظ الملفات:")
print(f"   fraud_model.pkl      ← النموذج الأفضل ({best_name})")
print(f"   scaler.pkl           ← أداة تطبيع الأرقام")
print(f"   label_encoder.pkl    ← أداة ترميز نوع التاجر")
print(f"   feature_names.pkl    ← أسماء الميزات")

# ============================================================
#  اختبار سريع للنموذج
# ============================================================
print("\n" + "=" * 55)
print("  🧪 اختبار سريع على معاملة جديدة")
print("=" * 55)

def predict_transaction(amount, hour, merchant, foreign, location_mismatch,
                         device_trust, velocity, age):
    model  = joblib.load('fraud_model.pkl')
    sc     = joblib.load('scaler.pkl')
    enc    = joblib.load('label_encoder.pkl')
    feats  = joblib.load('feature_names.pkl')

    merchant_enc = enc.transform([merchant])[0]
    row = np.array([[amount, hour, merchant_enc, foreign,
                     location_mismatch, device_trust, velocity, age]])
    row_sc = sc.transform(row)
    pred   = model.predict(row_sc)[0]
    proba  = model.predict_proba(row_sc)[0][1]
    return pred, proba

# مثال 1: معاملة طبيعية
pred, prob = predict_transaction(
    amount=50.0, hour=14, merchant='Grocery',
    foreign=0, location_mismatch=0, device_trust=85, velocity=2, age=35
)
print(f"\nمعاملة 1 (متوقع طبيعية):")
print(f"   النتيجة    : {'❌ احتيالية' if pred==1 else '✅ طبيعية'}")
print(f"   احتمال الاحتيال: {prob*100:.1f}%")

# مثال 2: معاملة مشبوهة
pred, prob = predict_transaction(
    amount=1400.0, hour=3, merchant='Electronics',
    foreign=1, location_mismatch=1, device_trust=10, velocity=15, age=22
)
print(f"\nمعاملة 2 (متوقع احتيالية):")
print(f"   النتيجة    : {'❌ احتيالية' if pred==1 else '✅ طبيعية'}")
print(f"   احتمال الاحتيال: {prob*100:.1f}%")

print("\n" + "=" * 55)
print("  🎉 انتهت جميع المراحل بنجاح!")
print("  الخطوة التالية: بناء واجهة Flask")
print("=" * 55)
