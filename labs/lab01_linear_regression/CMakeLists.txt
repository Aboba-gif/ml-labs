# Конфигурация сборки для Lab 01: Linear Regression.
# Copyright (C) 2024 ML Labs Project
#
# ОПИСАНИЕ:
# =========
# Реализация линейной регрессии с градиентным спуском.
# Включает: SGD, Mini-batch GD, Adam optimizer.
#
# АЛГОРИТМИЧЕСКАЯ СЛОЖНОСТЬ:
# - Training: O(n * m * epochs) где n - samples, m - features
# - Prediction: O(m) для одного примера
# - Memory: O(m) для весов + O(batch_size * m) для градиентов
#
# ОПТИМИЗАЦИИ:
# - Векторизация через Eigen BLAS
# - OpenMP параллелизация batch обработки
# - Cache-friendly data layout

# set - установка значения переменной в текущей области видимости.
# СИГНАТУРА: set(<variable> <value>... [PARENT_SCOPE | CACHE <type> <docstring> [FORCE]])
# ПАРАМЕТРЫ:
#   variable - имя переменной (регистрозависимое)
#   value - одно или несколько значений (создают список через точку с запятой)
#   PARENT_SCOPE - установить переменную в родительской области видимости
#   CACHE <type> <docstring> - создать кэшированную переменную
#   FORCE - перезаписать существующую CACHE переменную
# ТИПЫ CACHE ПЕРЕМЕННЫХ:
#   BOOL - булево значение (ON/OFF)
#   FILEPATH - путь к файлу
#   PATH - путь к директории  
#   STRING - произвольная строка
#   INTERNAL - внутренняя переменная (не показывается в GUI)
# ОБЛАСТЬ ВИДИМОСТИ: Локальная для текущей функции/директории
# ВЕРСИЯ: Базовая с CMake 2.6, PARENT_SCOPE с 2.8.3
set(LAB_NAME lab01_linear_regression)

# Создание исполняемого файла лабы

# add_executable - создание исполняемого файла (программы).
# СИГНАТУРА: add_executable(<name> [WIN32] [MACOSX_BUNDLE]
#                           [EXCLUDE_FROM_ALL]
#                           [source1] [source2 ...])
#           add_executable(<name> IMPORTED [GLOBAL])
#           add_executable(<name> ALIAS <target>)
# ПАРАМЕТРЫ:
#   name - имя создаваемого исполняемого файла
#   WIN32 - Windows приложение с WinMain вместо main (подавляет консоль)
#   MACOSX_BUNDLE - создать macOS .app bundle
#   EXCLUDE_FROM_ALL - не включать в цель ALL (не собирается по умолчанию)
#   source - исходные файлы (можно добавить позже через target_sources)
#   IMPORTED - импортированный исполняемый файл из внешнего проекта
#   GLOBAL - сделать IMPORTED таргет видимым глобально
#   ALIAS - создать псевдоним для существующего таргета
# СОЗДАЁТ: Таргет типа EXECUTABLE
# ВЕРСИЯ: Базовая с CMake 2.6, ALIAS с 2.8.11, IMPORTED расширения с 3.11
# ${LAB_NAME} - подстановка значения переменной LAB_NAME
add_executable(ml_labs_${LAB_NAME})

# Источники лабы

# target_sources - добавление исходных файлов к существующему таргету.
# СИГНАТУРА: target_sources(<target>
#                           <INTERFACE|PUBLIC|PRIVATE> [items1...]
#                           [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...]
#                           [FILE_SET <set> [TYPE <type>] [BASE_DIRS <dirs>...] 
#                            FILES <files>...])
# ПАРАМЕТРЫ:
#   target - имя цели к которой добавляются файлы
#   INTERFACE - файлы только для потребителей таргета (header-only библиотеки)
#   PUBLIC - файлы для таргета и его потребителей
#   PRIVATE - файлы только для компиляции самого таргета
#   FILE_SET - именованный набор файлов (CMake 3.23+)
#   TYPE - тип набора файлов (HEADERS, CXX_MODULES)
#   BASE_DIRS - базовые директории для относительных путей при установке
#   FILES - список файлов в наборе
# СЕМАНТИКА ВИДИМОСТИ:
#   PRIVATE - файлы компилируются только в этот таргет
#   PUBLIC - файлы компилируются в таргет и доступны зависимым таргетам
#   INTERFACE - файлы не компилируются в таргет, только в зависимые
# ВЕРСИЯ: Доступна с CMake 3.1, FILE_SET с 3.23
target_sources(ml_labs_${LAB_NAME} PRIVATE
    main.cpp
    src/gradient_descent.cpp
    src/data_loader.cpp
    src/visualization.cpp
)

# Заголовки лабы (если есть публичные компоненты)

# if - условный оператор для ветвления логики.
# СИГНАТУРА: if(<condition>) ... elseif(<condition>) ... else() ... endif()
# УСЛОВИЕ EXISTS: проверка существования файла или директории
# СИНТАКСИС: if(EXISTS path-to-file-or-directory)
# СЕМАНТИКА: Возвращает TRUE если путь существует в файловой системе
# ДЕТАЛИ: 
#   - Работает с абсолютными и относительными путями
#   - Для символических ссылок проверяет саму ссылку, не цель
#   - Проверка происходит во время configure, не во время сборки
# CMAKE_CURRENT_SOURCE_DIR - автоматическая переменная с путём к текущей исходной директории
# ОПРЕДЕЛЕНИЕ: Абсолютный путь к директории где находится обрабатываемый CMakeLists.txt
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    # target_include_directories - установка путей поиска заголовочных файлов для таргета.
    # СИГНАТУРА: target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
    #                                        <INTERFACE|PUBLIC|PRIVATE> [items1...]
    #                                        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
    # ПАРАМЕТРЫ:
    #   target - имя цели для которой устанавливаются пути
    #   SYSTEM - пометить директории как системные (подавляет предупреждения компилятора)
    #   AFTER|BEFORE - добавить пути в конец или начало списка (по умолчанию AFTER)
    #   INTERFACE - пути только для потребителей таргета
    #   PUBLIC - пути для таргета и его потребителей
    #   PRIVATE - пути только для компиляции самого таргета
    #   items - пути к директориям с заголовками
    # ГЕНЕРАТОРНЫЕ ВЫРАЖЕНИЯ:
    #   $<BUILD_INTERFACE:...> - путь активен только при сборке
    #   $<INSTALL_INTERFACE:...> - путь активен только после установки
    #   $<$<CONFIG:Debug>:...> - путь только для Debug конфигурации
    # ЭФФЕКТ: Добавляет -I<path> для GCC/Clang или /I<path> для MSVC
    # ВЕРСИЯ: Доступна с CMake 2.8.11
    target_include_directories(ml_labs_${LAB_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
# endif - обязательное завершение блока if
# СЕМАНТИКА: Каждому if() должен соответствовать endif()
endif()

# Зависимости

# target_link_libraries - связывание таргета с библиотеками и установка зависимостей.
# СИГНАТУРА: target_link_libraries(<target>
#                                   <PRIVATE|PUBLIC|INTERFACE> <item>...
#                                   [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
# ПАРАМЕТРЫ:
#   target - имя цели для линковки
#   PRIVATE - библиотеки только для линковки с таргетом
#   PUBLIC - библиотеки для таргета и его потребителей  
#   INTERFACE - библиотеки только для потребителей таргета
#   item - библиотека, таргет, путь к файлу или флаг линкера
# ФОРМАТЫ ITEM:
#   target_name - CMake таргет (предпочтительно)
#   namespace::target - таргет с namespace (современный стиль)
#   -lfoo - флаг линкера напрямую
#   /path/to/lib.a - полный путь к файлу библиотеки
#   debug/optimized/general - конфигурационные ключевые слова
# СЕМАНТИКА:
#   PRIVATE - символы библиотеки не экспортируются из таргета
#   PUBLIC - символы могут быть переэкспортированы
#   INTERFACE - для header-only библиотек
# ТРАНЗИТИВНОСТЬ: PUBLIC и INTERFACE зависимости наследуются потребителями
# ВЕРСИЯ: Базовая с CMake 2.6, INTERFACE/PUBLIC/PRIVATE с 2.8.11
target_link_libraries(ml_labs_${LAB_NAME} PRIVATE
    ml_labs::core      # Основная библиотека
    Eigen3::Eigen      # Линейная алгебра
    fmt::fmt           # Форматированный вывод
    Threads::Threads   # Многопоточность
)

# Применение project-wide настроек

# set_project_warnings - пользовательская функция из CompilerWarnings.cmake
# СИГНАТУРА: set_project_warnings(target_name [WARNING_LEVEL level])
# ПАРАМЕТРЫ:
#   target_name - имя таргета для применения предупреждений
#   WARNING_LEVEL - INTERFACE|PUBLIC|PRIVATE (по умолчанию PRIVATE)
# ЭФФЕКТ: 
#   - Устанавливает максимально строгие предупреждения компилятора
#   - Применяет -Wall -Wextra -Wpedantic и дополнительные флаги
#   - В Debug превращает предупреждения в ошибки (-Werror)
# ДЕТАЛИ: Использует генераторные выражения для поддержки разных компиляторов
set_project_warnings(ml_labs_${LAB_NAME})

# Оптимизации для Release

# CMAKE_BUILD_TYPE - глобальная переменная с типом конфигурации сборки
# ВОЗМОЖНЫЕ ЗНАЧЕНИЯ:
#   Debug - отладочная сборка (-g, без оптимизаций)
#   Release - релизная сборка (-O3, без отладочной информации)
#   RelWithDebInfo - релиз с отладочной информацией (-O2 -g)
#   MinSizeRel - минимальный размер (-Os)
# STREQUAL - оператор точного сравнения строк
# СИНТАКСИС: <variable> STREQUAL <string>
# СЕМАНТИКА: Регистрозависимое побайтовое сравнение
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # target_compile_options - добавление опций компиляции к таргету.
    # СИГНАТУРА: target_compile_options(<target> [BEFORE]
    #                                   <INTERFACE|PUBLIC|PRIVATE> [items1...]
    #                                   [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
    # ПАРАМЕТРЫ:
    #   target - имя цели
    #   BEFORE - добавить опции в начало списка (по умолчанию в конец)
    #   INTERFACE|PUBLIC|PRIVATE - область видимости опций
    #   items - флаги компилятора или генераторные выражения
    # ФЛАГИ ОПТИМИЗАЦИИ GCC/CLANG:
    #   -march=native - использовать все инструкции текущего CPU (AVX, SSE, etc.)
    #   -mtune=native - оптимизировать планирование инструкций для текущего CPU
    #   -ffast-math - агрессивные оптимизации float (нарушает IEEE 754)
    #   -funroll-loops - разворачивание циклов для уменьшения overhead
    # ПРЕДУПРЕЖДЕНИЕ: -march=native делает бинарник непереносимым
    # ВЕРСИЯ: Доступна с CMake 2.8.12
    target_compile_options(ml_labs_${LAB_NAME} PRIVATE
        -march=native      # Использовать инструкции текущего CPU
        -mtune=native      # Оптимизировать для текущего CPU
        -ffast-math        # Агрессивные float оптимизации
        -funroll-loops     # Разворачивание циклов
    )
endif()

# Установка имени выходного файла

# set_target_properties - установка свойств таргета.
# СИГНАТУРА: set_target_properties(target1 target2 ...
#                                   PROPERTIES prop1 value1
#                                   [prop2 value2] ...)
# ПАРАМЕТРЫ:
#   target - один или несколько таргетов
#   PROPERTIES - ключевое слово начала списка свойств
#   prop value - пары свойство-значение
# РАСПРОСТРАНЁННЫЕ СВОЙСТВА:
#   OUTPUT_NAME - имя выходного файла без расширения
#   PREFIX - префикс имени файла (lib для библиотек)
#   SUFFIX - суффикс/расширение файла (.so, .exe)
#   RUNTIME_OUTPUT_DIRECTORY - директория для исполняемых файлов
#   LIBRARY_OUTPUT_DIRECTORY - директория для shared библиотек
#   ARCHIVE_OUTPUT_DIRECTORY - директория для static библиотек
#   VERSION - версия библиотеки (создаёт симлинки .so.1.2.3)
#   SOVERSION - версия SONAME для shared библиотек
#   PUBLIC_HEADER - публичные заголовки для установки
#   POSITION_INDEPENDENT_CODE - включить PIC
#   CXX_STANDARD - стандарт C++ для таргета
#   CXX_STANDARD_REQUIRED - требовать точный стандарт
#   COMPILE_OPTIONS - дополнительные опции компиляции
#   COMPILE_DEFINITIONS - макроопределения препроцессора
#   LINK_OPTIONS - опции линковщика
#   LINK_FLAGS - флаги линковщика (устарело, используйте LINK_OPTIONS)
#   FOLDER - папка в IDE (Visual Studio, Xcode)
#   VS_DEBUGGER_WORKING_DIRECTORY - рабочая директория отладчика VS
# CMAKE_BINARY_DIR - автоматическая переменная с путём к корню build директории
# ВЕРСИЯ: Доступна с CMake 2.6
set_target_properties(ml_labs_${LAB_NAME} PROPERTIES
    OUTPUT_NAME ${LAB_NAME}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Копирование данных для лабы

# file - операции с файловой системой.
# СИГНАТУРА для GLOB: file(GLOB <variable>
#                          [LIST_DIRECTORIES true|false]
#                          [RELATIVE <path>]
#                          [CONFIGURE_DEPENDS]
#                          <globbing-expressions>...)
# ПАРАМЕТРЫ:
#   GLOB - операция поиска файлов по шаблону (wildcard)
#   variable - переменная для сохранения списка найденных файлов
#   LIST_DIRECTORIES - включать директории в результат (по умолчанию true)
#   RELATIVE path - возвращать относительные пути от указанной директории
#   CONFIGURE_DEPENDS - перезапускать CMake при изменении файлов (CMake 3.12+)
#   globbing-expressions - шаблоны поиска
# ШАБЛОНЫ (WILDCARDS):
#   * - любая последовательность символов (кроме /)
#   ? - любой одиночный символ
#   [abc] - любой символ из набора
#   [a-z] - любой символ из диапазона
#   ** - рекурсивный поиск (GLOB_RECURSE)
# ПРЕДУПРЕЖДЕНИЕ: GLOB не отслеживает добавление/удаление файлов автоматически
# РЕКОМЕНДАЦИЯ: Явно перечислять файлы вместо GLOB для надёжности
file(GLOB DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/*.csv)

if(DATA_FILES)
    # file - операции с файловой системой.
    # СИГНАТУРА для COPY: file(COPY <files>... DESTINATION <dir>
    #                          [NO_SOURCE_PERMISSIONS]
    #                          [USE_SOURCE_PERMISSIONS]
    #                          [FILE_PERMISSIONS <permissions>...]
    #                          [DIRECTORY_PERMISSIONS <permissions>...]
    #                          [FOLLOW_SYMLINK_CHAIN]
    #                          [FILES_MATCHING]
    #                          [[PATTERN <pattern> | REGEX <regex>]
    #                           [EXCLUDE] [PERMISSIONS <permissions>...]] [...])
    # ПАРАМЕТРЫ:
    #   COPY - операция копирования файлов
    #   files - список файлов/директорий для копирования
    #   DESTINATION - целевая директория
    #   NO_SOURCE_PERMISSIONS - не копировать права доступа
    #   USE_SOURCE_PERMISSIONS - копировать исходные права (по умолчанию)
    #   FILE_PERMISSIONS - установить права для файлов
    #   DIRECTORY_PERMISSIONS - установить права для директорий
    #   FOLLOW_SYMLINK_CHAIN - следовать по цепочке симлинков
    #   FILES_MATCHING - копировать только файлы соответствующие паттерну
    #   PATTERN/REGEX - фильтры для выборочного копирования
    #   EXCLUDE - исключить файлы по паттерну
    # ПРАВА ДОСТУПА (UNIX):
    #   OWNER_READ, OWNER_WRITE, OWNER_EXECUTE
    #   GROUP_READ, GROUP_WRITE, GROUP_EXECUTE
    #   WORLD_READ, WORLD_WRITE, WORLD_EXECUTE
    # ДЕТАЛИ: Копирование происходит во время configure, не во время сборки
    # АЛЬТЕРНАТИВА: configure_file() для одиночных файлов с подстановкой
    # ВЕРСИЯ: Доступна с CMake 2.8
    file(COPY ${DATA_FILES} 
         DESTINATION ${CMAKE_BINARY_DIR}/labs/${LAB_NAME}/data)
endif()

# Unit тесты для лабы (если есть)

# ML_LABS_BUILD_TESTS - глобальная опция из корневого CMakeLists.txt
# AND - логический оператор "И"
if(ML_LABS_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    # Создание исполняемого файла для тестов
    add_executable(test_${LAB_NAME}
        tests/test_gradient_descent.cpp
        tests/test_data_loader.cpp
    )
    
    # GTest::gtest_main - таргет Google Test с функцией main()
    # ДЕТАЛИ: gtest_main предоставляет стандартную main() которая запускает все тесты
    target_link_libraries(test_${LAB_NAME} PRIVATE
        ml_labs::core
        GTest::gtest_main
    )
    
    # Регистрация тестов в CTest
    
    # include - подключение CMake модуля или файла.
    # GoogleTest - модуль для интеграции с Google Test framework
    # ПРЕДОСТАВЛЯЕТ: gtest_add_tests(), gtest_discover_tests()
    include(GoogleTest)
    
    # gtest_discover_tests - автоматическое обнаружение тестов в исполняемом файле.
    # СИГНАТУРА: gtest_discover_tests(target
    #                                  [EXTRA_ARGS args...]
    #                                  [WORKING_DIRECTORY dir]
    #                                  [TEST_PREFIX prefix]
    #                                  [TEST_SUFFIX suffix]
    #                                  [TEST_FILTER filter]
    #                                  [NO_PRETTY_TYPES]
    #                                  [NO_PRETTY_VALUES]
    #                                  [PROPERTIES name1 value1...]
    #                                  [TEST_LIST var]
    #                                  [DISCOVERY_TIMEOUT seconds]
    #                                  [XML_OUTPUT_DIR dir]
    #                                  [DISCOVERY_MODE <POST_BUILD|PRE_TEST>])
    # ПАРАМЕТРЫ:
    #   target - исполняемый файл с тестами
    #   EXTRA_ARGS - дополнительные аргументы для запуска тестов
    #   WORKING_DIRECTORY - рабочая директория для тестов
    #   TEST_PREFIX - префикс для имён тестов в CTest
    #   TEST_SUFFIX - суффикс для имён тестов
    #   TEST_FILTER - фильтр Google Test (--gtest_filter)
    #   NO_PRETTY_TYPES - не украшать имена типов
    #   NO_PRETTY_VALUES - не украшать значения
    #   PROPERTIES - свойства для тестов CTest
    #   TEST_LIST - переменная для списка обнаруженных тестов
    #   DISCOVERY_TIMEOUT - таймаут обнаружения тестов
    #   XML_OUTPUT_DIR - директория для XML отчётов
    #   DISCOVERY_MODE - когда обнаруживать тесты (сборка или запуск)
    # СВОЙСТВО LABELS: метки для группировки тестов
    # ИСПОЛЬЗОВАНИЕ: ctest -L lab01 запустит только тесты с меткой lab01
    # ВЕРСИЯ: Доступна с CMake 3.10
    gtest_discover_tests(test_${LAB_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/labs/${LAB_NAME}
        PROPERTIES LABELS "lab01;unit"
    )
endif()

# Benchmark для производительности
if(ML_LABS_BUILD_BENCHMARKS)
    add_executable(bench_${LAB_NAME}
        benchmarks/bench_training.cpp
    )
    
    # benchmark::benchmark_main - Google Benchmark с main()
    # ДЕТАЛИ: benchmark_main предоставляет main() с парсингом аргументов командной строки
    target_link_libraries(bench_${LAB_NAME} PRIVATE
        ml_labs::core
        benchmark::benchmark_main
    )
endif()

# Генерация отчёта о лабе

# add_custom_command - добавление пользовательской команды сборки.
# СИГНАТУРА для POST_BUILD: 
#   add_custom_command(TARGET <target>
#                      PRE_BUILD | PRE_LINK | POST_BUILD
#                      COMMAND command1 [ARGS] [args1...]
#                      [COMMAND command2 [ARGS] [args2...] ...]
#                      [BYPRODUCTS files...]
#                      [WORKING_DIRECTORY dir]
#                      [COMMENT comment]
#                      [VERBATIM] [USES_TERMINAL]
#                      [COMMAND_EXPAND_LISTS])
# ПАРАМЕТРЫ:
#   TARGET - таргет к которому привязывается команда
#   PRE_BUILD - выполнить перед сборкой (только Visual Studio)
#   PRE_LINK - выполнить перед линковкой
#   POST_BUILD - выполнить после успешной сборки таргета
#   COMMAND - команда для выполнения
#   BYPRODUCTS - файлы создаваемые командой (для ninja)
#   WORKING_DIRECTORY - рабочая директория
#   COMMENT - описание для вывода
#   VERBATIM - точная передача аргументов без shell интерпретации
#   USES_TERMINAL - использовать терминал напрямую (CMake 3.2+)
#   COMMAND_EXPAND_LISTS - раскрывать списки в аргументах (CMake 3.8+)
# CMAKE_COMMAND - автоматическая переменная с путём к исполняемому файлу cmake
# ОПРЕДЕЛЕНИЕ: Полный путь к cmake используемому для конфигурации
# -E - встроенные кроссплатформенные команды CMake
# КОМАНДЫ cmake -E:
#   echo - вывод текста
#   copy - копирование файлов
#   copy_directory - копирование директорий
#   copy_if_different - копировать если отличается
#   remove - удаление файлов
#   remove_directory - удаление директорий
#   rename - переименование
#   compare_files - сравнение файлов
#   create_symlink - создание символической ссылки
#   create_hardlink - создание жёсткой ссылки
#   touch - обновление временной метки
#   touch_nocreate - обновить метку без создания
#   make_directory - создание директорий
#   md5sum - вычисление MD5
#   sha1sum - вычисление SHA1
#   sha256sum - вычисление SHA256
#   tar - работа с архивами
#   sleep - пауза в секундах
#   time - измерение времени выполнения
#   env - установка переменных окружения
#   server - запуск cmake-server (устарело)
#   capabilities - возможности cmake -E
# ВЕРСИЯ: Базовая с CMake 2.6, расширения команд добавлялись постепенно
add_custom_command(TARGET ml_labs_${LAB_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
    "Lab 01: Linear Regression built successfully"
    COMMAND ${CMAKE_COMMAND} -E echo
    "Run with: ${CMAKE_BINARY_DIR}/bin/${LAB_NAME}"
)
