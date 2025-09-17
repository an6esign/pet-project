# src/models/train.py
import json
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ARTIFACT_MODEL_PATH = "models/artifacts/best_text_router.joblib"
ARTIFACT_LABELMAP_PATH = "models/artifacts/label_map.json"
FIG_CM_PATH = "reports/figures/confusion_matrix.png"

TARGET_NAMES = ["Вопросы по Telegram Ads", "Баг/жалоба", "Дизайн", "Заказ", "Мусор"]

def get_dataset():
    texts = [
        # 0 – Вопросы по Telegram Ads
        "как работает автоматическое размещение в Telegram Ads?",
        "почему объявление не проходит модерацию?",
        "какие есть ограничения по тексту объявления?",
        "можно ли загрузить видео вместо баннера?",
        "как установить UTM-метки на ссылку?",
        "откуда берутся показы — это только ленты?",
        "можно ли таргетироваться по подписчикам конкурентов?",
        "что влияет на CPM в Telegram Ads?",
        "есть ли возможность ретаргетинга?",
        "как понять, в каком канале показалось объявление?",
        "можно ли выбрать конкретные Telegram-каналы для показа?",
        "какой минимальный бюджет для запуска рекламы?",
        "как пополнить счёт без карты?",
        "почему статистика не обновляется?",
        "как настроить отслеживание заявок из Telegram?",
        "что делать, если открутка остановилась на 10%?",
        "в чём разница между CPM и oCPM?",
        "можно ли менять креатив после запуска кампании?",
        "где посмотреть CTR по объявлениям?",
        "доступна ли аналитика по аудитории?",

        # 1 – Жалоба / баг
        "у меня не загружается страница",
        "пропал весь мой текст",
        "не работает кнопка сохранить",
        "почему всё зависает?",
        "сайт выдает ошибку 500",
        "не могу авторизоваться",
        "не проходит оплата",
        "бот не отвечает",
        "появляется пустое окно",
        "у меня крашится приложение",
        "поле не заполняется",
        "ничего не отображается",
        "сайт стал тормозить",
        "не приходят уведомления",
        "приложение закрывается само",
        "выдаёт ошибку при нажатии",
        "ничего не происходит после клика",
        "у меня пропал доступ",
        "не могу перейти по ссылке",
        "ошибка при загрузке файла",

        # 2 – Дизайн-запросы
        "можно ли сделать кнопку выше?",
        "давайте перенесём логотип налево",
        "сделайте текст крупнее",
        "цвет слишком блеклый",
        "добавьте тень к заголовку",
        "можно убрать отступы?",
        "сделайте кнопку круглее",
        "давайте заменим иконку",
        "нужно выровнять по центру",
        "шрифт слишком мелкий",
        "надо сделать больше контраст",
        "добавьте ховер к ссылке",
        "можно градиент вместо фона?",
        "логотип расплывается",
        "кнопка не бросается в глаза",
        "давайте уберём нижний бордер",
        "надо сделать заголовок жирным",
        "плохо видно текст на фоне",
        "добавьте пустое место сверху",
        "сделайте интерфейс воздушнее",

        # 3 – Оформление заказа
        "как оплатить доступ на месяц?",
        "хочу подписаться",
        "можно купить тариф PRO?",
        "оформите мне заказ, пожалуйста",
        "сколько стоит обучение?",
        "хочу продлить подписку",
        "отправьте реквизиты для оплаты",
        "куда переводить деньги?",
        "я готов оплатить",
        "можно перейти на PRO?",
        "как подключить модуль?",
        "хочу доступ к полному функционалу",
        "можно оформить заказ на компанию?",
        "пришлите счёт",
        "хочу оплатить картой",
        "могу ли я оплатить позже?",
        "как перейти на годовой тариф?",
        "добавьте меня в список участников",
        "можно оплатить в рассрочку?",
        "где кнопка «оформить заказ»?",

        # 4 – Мусор / неинформативное
        "аааа",
        "чтоооооо?",
        "???",
        "плиз помогите",
        "лол",
        "это вообще что такое?",
        "ну как бы…",
        "ммм непонятно",
        "я не понял ничего",
        "что делать?",
        "дайте что-нибудь",
        "ничего не понял",
        "помогите",
        "ха-ха-ха",
        "зачем так сложно?",
        "вот это прикол",
        "что за ерунда?",
        "не понял",
        "аааа объясните",
        "вы чё серьёзно?"
    ]
    labels = [0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20
    return pd.DataFrame({"text": texts, "label": labels})

def train():
    df = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, stratify=df['label'], random_state=0
    )

    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    param_grid = [
        # TF‑IDF + LogReg
        {'vectorizer': [TfidfVectorizer(ngram_range=(1,1), min_df=1),
                        TfidfVectorizer(ngram_range=(1,2), min_df=2)],
         'clf': [LogisticRegression(max_iter=2000)],
         'clf__C': [0.3, 1.0, 3.0]},
        # TF‑IDF + LinearSVC (без вероятностей)
        {'vectorizer': [TfidfVectorizer(ngram_range=(1,1), min_df=1),
                        TfidfVectorizer(ngram_range=(1,2), min_df=2)],
         'clf': [LinearSVC()],
         'clf__C': [0.3, 1.0, 3.0]}
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv,
                        scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    print('\nЛучшие параметры:', grid.best_params_)
    y_pred = best.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Macro‑F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print('\nClassification report:\n',
          classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    # Confusion matrix (сохранить в reports/figures)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.xlabel("Предсказано")
    plt.ylabel("Истинно")
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.savefig(FIG_CM_PATH, dpi=150)
    plt.close()
    print(f"Сохранено изображение: {FIG_CM_PATH}")

    # Сохранение артефактов
    joblib.dump(best, ARTIFACT_MODEL_PATH)
    label_map = {i: name for i, name in enumerate(TARGET_NAMES)}
    with open(ARTIFACT_LABELMAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Сохранены артефакты: {ARTIFACT_MODEL_PATH}, {ARTIFACT_LABELMAP_PATH}")

if __name__ == "__main__":
    train()
