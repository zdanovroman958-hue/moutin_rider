[app]

# Информация о приложении (видно в Android)
title = Горный байкер
package.name = mountainrider
package.domain = com.gorobiker

source.dir = .
source.include_exts = py,png,txt
source.exclude_dirs = __pycache__, .git, .github

# Версия для Android
version = 1.0.0

# Библиотеки, которые собираются в APK.
# Игра использует pygame-ce; в python-for-android рецепт называется pygame.
# numpy и pymunk - рецепты есть в p4a.
requirements = python3==3.9.18,hostpython3==3.9.18,pygame,pymunk,numpy==v2.0.1

# Внешний вид и ориентация
orientation = landscape
fullscreen = 1
android.permissions = VIBRATE
android.entrypoint = main.py
android.accept_sdk_license = True

# (Важно) иконка/сплэш опциональны. Без них соберётся APK-заглушка с дефолтным значком.
# android.icon.filename = %(source.dir)s/icon.png

# API и SDK. 33 - Android 13, minapi 24 - Android 7.0.
android.api = 33
android.minapi = 24

# NDK можно не указывать версию - buildozer подберёт совместимую с API 33.
28c

# Архитектуры телефонов (arm64 - современные, armeabi-v7a - старые)
android.archs = arm64-v8a

# Тип сборки: debug имеет отладочный значок и подпись, подходит для теста.
# Релиз (release) требует ключи для подписи - см. README_BUILD_APK.md.
android.console = 0
android.logcat = 1

[buildozer]

log_level = 2
warn_on_root = 1

# Кэш и целевая папка buildozer
build_dir = ./.buildozer
bin_dir = ./bin