# Конфигурация модульных и интеграционных тестов.
# Copyright (C) 2024 ML Labs Project
#
# ТЕСТОВАЯ СТРАТЕГИЯ:
# ===================
# - Unit tests: изолированное тестирование компонентов
# - Integration tests: тестирование взаимодействия
# - Property-based tests: проверка инвариантов
# - Benchmark tests: регрессии производительности
#
# МЕТРИКИ КАЧЕСТВА:
# - Line coverage > 80%
# - Branch coverage > 70%  
# - Mutation score > 60%
#
# ФРЕЙМВОРКИ:
# - Google Test для unit/integration
# - Google Benchmark для производительности
# - RapidCheck для property-based testing

# Поиск тестовых фреймворков

# find_package - поиск и загрузка настроек внешнего пакета.
# СИГНАТУРА: find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
#                         [REQUIRED] [[COMPONENTS] [components...]]
#                         [OPTIONAL_COMPONENTS components...]
#                         [CONFIG|NO_MODULE]
#                         [NAMES name1 [name2 ...]]
#                         [CONFIGS config1 [config2 ...]]
#                         [HINTS path1 [path2 ... ]]
#                         [PATHS path1 [path2 ... ]]
#                         [REGISTRY_VIEW <64|32|64_32|32_64|HOST|TARGET|BOTH>]
#                         [GLOBAL]
#                         [NO_POLICY_SCOPE]
#                         [BYPASS_PROVIDER])
# ПАРАМЕТРЫ:
#   PackageName - имя пакета для поиска
#   version - минимальная требуемая версия (major[.minor[.patch[.tweak]]])
#   EXACT - требовать точное совпадение версии
#   QUIET - не выводить сообщения (кроме ошибок)
#   MODULE - использовать только Module mode (Find<Package>.cmake)
#   REQUIRED - прерывание с ошибкой если пакет не найден
#   COMPONENTS - обязательные компоненты пакета
#   OPTIONAL_COMPONENTS - опциональные компоненты
#   CONFIG/NO_MODULE - использовать только Config mode (<Package>Config.cmake)
# РЕЖИМЫ ПОИСКА:
#   Module mode - ищет Find<Package>.cmake в CMAKE_MODULE_PATH
#   Config mode - ищет <Package>Config.cmake или <package>-config.cmake
# СОЗДАЁТ ПЕРЕМЕННЫЕ:
#   <PackageName>_FOUND - TRUE если пакет найден
#   <PackageName>_VERSION - версия найденного пакета
#   <PackageName>_VERSION_MAJOR/MINOR/PATCH/TWEAK - компоненты версии
#   <PackageName>_INCLUDE_DIRS - директории заголовков
#   <PackageName>_LIBRARIES - библиотеки для линковки
#   <PackageName>_DEFINITIONS - флаги препроцессора
# ИМПОРТИРОВАННЫЕ ТАРГЕТЫ:
#   <PackageName>::<ComponentName> - современный способ использования
# ВЕРСИЯ: Базовая с CMake 2.6, расширения в 3.x
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Включение CTest для интеграции с CMake

# include - подключение CMake модуля или файла.
# СИГНАТУРА: include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>] 
#                    [NO_POLICY_SCOPE])
# ПАРАМЕТРЫ:
#   file|module - путь к файлу или имя модуля
#   OPTIONAL - не выдавать ошибку если файл/модуль не найден
#   RESULT_VARIABLE - переменная для сохранения полного пути найденного файла
#   NO_POLICY_SCOPE - не создавать новую область для политик CMake
# ПОИСК МОДУЛЕЙ:
#   1. CMAKE_MODULE_PATH - пользовательские пути
#   2. <CMAKE_ROOT>/Modules - встроенные модули CMake
# CTest - встроенный модуль для интеграции с системой тестирования CTest
# ПРЕДОСТАВЛЯЕТ:
#   enable_testing() - включение тестирования
#   add_test() - добавление тестов
#   set_tests_properties() - установка свойств тестов
#   Переменные BUILD_TESTING, DART_TESTING_TIMEOUT и др.
# ВЕРСИЯ: Доступен с CMake 2.8
include(CTest)

# GoogleTest - модуль для интеграции с Google Test framework
# ПРЕДОСТАВЛЯЕТ:
#   gtest_add_tests() - ручное добавление тестов (устарело)
#   gtest_discover_tests() - автоматическое обнаружение тестов
# ВЕРСИЯ: Доступен с CMake 3.9
include(GoogleTest)

# Основной тестовый исполняемый файл

# add_executable - создание исполняемого файла.
# СИГНАТУРА: add_executable(<name> [WIN32] [MACOSX_BUNDLE]
#                           [EXCLUDE_FROM_ALL]
#                           [source1] [source2 ...])
# ПАРАМЕТРЫ:
#   name - имя создаваемого исполняемого файла
#   WIN32 - Windows GUI приложение (WinMain вместо main)
#   MACOSX_BUNDLE - macOS .app bundle
#   EXCLUDE_FROM_ALL - не включать в цель ALL
#   source - исходные файлы (можно добавить позже через target_sources)
# СОЗДАЁТ: Таргет типа EXECUTABLE
# ДЕТАЛИ: Без исходников создаёт таргет для последующего добавления через target_sources
# ВЕРСИЯ: Базовая с CMake 2.6
add_executable(ml_labs_tests)

# Unit тесты core библиотеки

# target_sources - добавление исходных файлов к существующему таргету.
# СИГНАТУРА: target_sources(<target>
#                           <INTERFACE|PUBLIC|PRIVATE> [items1...]
#                           [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...]
#                           [FILE_SET <set> [TYPE <type>] [BASE_DIRS <dirs>...] 
#                            FILES <files>...])
# ПАРАМЕТРЫ:
#   target - имя цели к которой добавляются файлы
#   INTERFACE - файлы только для потребителей таргета
#   PUBLIC - файлы для таргета и его потребителей
#   PRIVATE - файлы только для компиляции самого таргета
# СЕМАНТИКА ВИДИМОСТИ:
#   PRIVATE - файлы компилируются только в этот таргет
#   PUBLIC - файлы видны зависимым таргетам (транзитивность)
#   INTERFACE - файлы не компилируются в таргет, только в зависимые
# ОРГАНИЗАЦИЯ КОДА: Группировка по функциональности с комментариями
# ВЕРСИЯ: Доступна с CMake 3.1
target_sources(ml_labs_tests PRIVATE
    # Тесты тензорных операций
    unit/test_tensor.cpp
    unit/test_tensor_ops.cpp
    unit/test_tensor_broadcasting.cpp
    
    # Тесты автодифференцирования
    unit/test_autodiff.cpp
    unit/test_gradient_tape.cpp
    unit/test_backprop.cpp
    
    # Тесты оптимизаторов
    unit/test_sgd.cpp
    unit/test_adam.cpp
    unit/test_rmsprop.cpp
    
    # Тесты функций потерь
    unit/test_mse_loss.cpp
    unit/test_cross_entropy.cpp
    unit/test_focal_loss.cpp
    
    # Тесты активаций
    unit/test_relu.cpp
    unit/test_sigmoid.cpp
    unit/test_softmax.cpp
)

# Integration тесты
target_sources(ml_labs_tests PRIVATE
    integration/test_training_pipeline.cpp
    integration/test_model_serialization.cpp
    integration/test_data_pipeline.cpp
    integration/test_distributed_training.cpp
)

# Зависимости тестов

# target_link_libraries - связывание таргета с библиотеками.
# СИГНАТУРА: target_link_libraries(<target>
#                                   <PRIVATE|PUBLIC|INTERFACE> <item>...
#                                   [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
# ПАРАМЕТРЫ:
#   target - имя цели для линковки
#   PRIVATE - библиотеки только для линковки с таргетом
#   PUBLIC - библиотеки для таргета и его потребителей
#   INTERFACE - библиотеки только для потребителей
#   item - библиотека, таргет или флаг линкера
# ИМПОРТИРОВАННЫЕ ТАРГЕТЫ:
#   GTest::gtest - основная библиотека Google Test
#   GTest::gtest_main - библиотека с функцией main()
#   GTest::gmock - Google Mock для создания mock объектов
# ТРАНЗИТИВНОСТЬ: PUBLIC и INTERFACE зависимости наследуются
# ВЕРСИЯ: Базовая с CMake 2.6, INTERFACE/PUBLIC/PRIVATE с 2.8.11
target_link_libraries(ml_labs_tests PRIVATE
    ml_labs::core
    GTest::gtest
    GTest::gtest_main
    GTest::gmock        # Для mocking
    Threads::Threads
)

# Include пути для тестов

# target_include_directories - установка путей поиска заголовков.
# СИГНАТУРА: target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
#                                        <INTERFACE|PUBLIC|PRIVATE> [items1...]
#                                        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
# ПАРАМЕТРЫ:
#   target - имя цели
#   SYSTEM - пометить как системные (подавляет предупреждения)
#   AFTER|BEFORE - добавить в конец или начало списка
#   INTERFACE|PUBLIC|PRIVATE - область видимости
#   items - пути к директориям
# CMAKE_CURRENT_SOURCE_DIR - путь к текущей исходной директории
# CMAKE_SOURCE_DIR - путь к корневой исходной директории проекта
# ДЕТАЛИ: Доступ к приватным заголовкам для white-box тестирования
# ВЕРСИЯ: Доступна с CMake 2.8.11
target_include_directories(ml_labs_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src  # Для тестирования приватных деталей
)

# Применение настроек проекта

# set_project_warnings - пользовательская функция из CompilerWarnings.cmake
# ЭФФЕКТ: Применяет строгие предупреждения компилятора к таргету
set_project_warnings(ml_labs_tests)

# Отключаем некоторые warnings для тестов

# target_compile_options - добавление опций компиляции.
# СИГНАТУРА: target_compile_options(<target> [BEFORE]
#                                   <INTERFACE|PUBLIC|PRIVATE> [items1...]
#                                   [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
# ГЕНЕРАТОРНЫЕ ВЫРАЖЕНИЯ:
#   $<$<CXX_COMPILER_ID:id>:...> - условное включение для компилятора
#   $<CXX_COMPILER_ID:GNU> - TRUE если компилятор GNU GCC
# ФЛАГИ КОМПИЛЯТОРА:
#   -Wno-gnu-zero-variadic-macro-arguments - отключить предупреждение о
#     вариадических макросах с нулём аргументов (специфично для GCC)
# ПРИЧИНА: Google Test использует вариадические макросы которые вызывают это предупреждение
# ВЕРСИЯ: Доступна с CMake 2.8.12
target_compile_options(ml_labs_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-gnu-zero-variadic-macro-arguments>
)

# Включение sanitizers для тестов

# if - условный оператор.
# ML_LABS_ENABLE_SANITIZERS - глобальная опция из корневого CMakeLists.txt
if(ML_LABS_ENABLE_SANITIZERS)
    # enable_sanitizers - пользовательская функция из Sanitizers.cmake
    # СИГНАТУРА: enable_sanitizers(target SANITIZERS sanitizer1 [sanitizer2 ...])
    # SANITIZERS:
    #   address - AddressSanitizer для обнаружения переполнений и use-after-free
    #   undefined - UndefinedBehaviorSanitizer для undefined behavior
    #   thread - ThreadSanitizer для data races (несовместим с address)
    #   memory - MemorySanitizer для неинициализированной памяти
    # ЭФФЕКТ: Добавляет -fsanitize=address -fsanitize=undefined и другие флаги
    enable_sanitizers(ml_labs_tests SANITIZERS address undefined)
# endif - завершение блока if
endif()

# Coverage для тестов
if(ML_LABS_ENABLE_COVERAGE)
    # enable_coverage - пользовательская функция из Coverage.cmake
    # ЭФФЕКТ: Добавляет --coverage и флаги для генерации .gcda/.gcno файлов
    enable_coverage(ml_labs_tests)
endif()

# Регистрация тестов в CTest

# gtest_discover_tests - автоматическое обнаружение Google Test тестов.
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
#   EXTRA_ARGS - дополнительные аргументы командной строки
#     --gtest_color=yes - цветной вывод в терминале
#   WORKING_DIRECTORY - рабочая директория для запуска тестов
#   TEST_PREFIX/SUFFIX - префикс/суффикс для имён тестов в CTest
#   TEST_FILTER - фильтр тестов (--gtest_filter)
#   PROPERTIES - свойства для тестов CTest:
#     TIMEOUT - максимальное время выполнения в секундах
#     LABELS - метки для группировки и фильтрации тестов
#     COST - относительная стоимость для планировщика
#     DEPENDS - зависимости от других тестов
#     ENVIRONMENT - переменные окружения
#     FIXTURES_SETUP/CLEANUP/REQUIRED - fixtures для тестов
#     PASS_REGULAR_EXPRESSION - regex для успешного прохождения
#     FAIL_REGULAR_EXPRESSION - regex для провала
#     SKIP_REGULAR_EXPRESSION - regex для пропуска
#     WILL_FAIL - тест должен провалиться
#     DISABLED - тест отключен
#     RUN_SERIAL - запускать последовательно
#     PROCESSORS - количество процессоров для теста
#     PROCESSOR_AFFINITY - привязка к процессорам
#     RESOURCE_LOCK - блокировка ресурсов
#     MEASUREMENT - измерения производительности
#   DISCOVERY_TIMEOUT - таймаут обнаружения тестов в секундах
#   XML_OUTPUT_DIR - директория для XML отчётов JUnit
#   DISCOVERY_MODE:
#     POST_BUILD - обнаружение после сборки (по умолчанию)
#     PRE_TEST - обнаружение перед запуском тестов
# ИСПОЛЬЗОВАНИЕ LABELS: ctest -L unit запустит только тесты с меткой "unit"
# ВЕРСИЯ: Доступна с CMake 3.10
gtest_discover_tests(ml_labs_tests
    EXTRA_ARGS --gtest_color=yes
    PROPERTIES 
        TIMEOUT 60
        LABELS "unit;core"
    DISCOVERY_TIMEOUT 10
)

# Benchmark тесты
add_executable(ml_labs_bench_tests
    benchmarks/bench_tensor_ops.cpp
    benchmarks/bench_autodiff.cpp
    benchmarks/bench_training.cpp
)

target_link_libraries(ml_labs_bench_tests PRIVATE
    ml_labs::core
    benchmark::benchmark      # Google Benchmark основная библиотека
    benchmark::benchmark_main # Google Benchmark с main()
)

# Custom target для запуска только unit тестов

# add_custom_target - создание пользовательской цели.
# СИГНАТУРА: add_custom_target(Name [ALL] [command1 [args1...]]
#                              [COMMAND command2 [args2...] ...]
#                              [DEPENDS depend depend ...]
#                              [BYPRODUCTS files...]
#                              [WORKING_DIRECTORY dir]
#                              [COMMENT comment]
#                              [JOB_POOL job_pool]
#                              [VERBATIM] [USES_TERMINAL]
#                              [COMMAND_EXPAND_LISTS]
#                              [SOURCES src1 [src2...]])
# ПАРАМЕТРЫ:
#   Name - имя создаваемой цели
#   ALL - включить в цель по умолчанию
#   COMMAND - команда для выполнения
#   DEPENDS - зависимости (файлы или таргеты)
#   BYPRODUCTS - файлы создаваемые командой
#   WORKING_DIRECTORY - рабочая директория
#   COMMENT - описание для вывода при сборке
#   VERBATIM - точная передача аргументов
#   USES_TERMINAL - использовать терминал напрямую
# CMAKE_CTEST_COMMAND - автоматическая переменная с путём к ctest
# ОПРЕДЕЛЕНИЕ: Путь к исполняемому файлу ctest той же версии что и cmake
# АРГУМЕНТЫ CTEST:
#   -L label - запустить только тесты с указанной меткой
#   -R regex - запустить тесты соответствующие regex
#   -E regex - исключить тесты соответствующие regex
#   --output-on-failure - показать вывод только для проваленных тестов
#   --verbose - подробный вывод
#   --quiet - минимальный вывод
#   --progress - показывать прогресс
#   --parallel n - запускать n тестов параллельно
#   --test-action test - только запустить тесты
#   --build-and-test - собрать и запустить
#   --rerun-failed - перезапустить проваленные
#   --repeat mode:n - повторить тесты n раз
#   --stop-on-failure - остановиться на первом провале
#   --timeout seconds - общий таймаут
#   --no-label-summary - не показывать сводку по меткам
# ВЕРСИЯ: Доступна с CMake 2.6
# Custom target для запуска только integration тестов  
add_custom_target(integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration --output-on-failure
    DEPENDS ml_labs_tests
    COMMENT "Running integration tests..."
)

# Custom target для запуска тестов с valgrind

# add_custom_target - создание пользовательской цели.
# СИГНАТУРА: add_custom_target(Name [ALL] [command1 [args1...]]
#                              [COMMAND command2 [args2...] ...]
#                              [DEPENDS depend depend ...]
#                              [BYPRODUCTS files...]
#                              [WORKING_DIRECTORY dir]
#                              [COMMENT comment]
#                              [JOB_POOL job_pool]
#                              [VERBATIM] [USES_TERMINAL]
#                              [COMMAND_EXPAND_LISTS]
#                              [SOURCES src1 [src2...]])
add_custom_target(memcheck_tests
    # CMAKE_CTEST_COMMAND - автоматическая переменная с путём к ctest
    # ОПРЕДЕЛЕНИЕ: Полный путь к исполняемому файлу ctest используемому с текущим cmake
    # АРГУМЕНТЫ CTEST ДЛЯ MEMCHECK:
    #   --overwrite - переопределение переменных CTest
    #   MemoryCheckCommand - команда для проверки памяти (valgrind, drmemory, cuda-memcheck)
    #   MemoryCheckCommandOptions - опции для команды проверки памяти
    #   -T memcheck - запустить тесты с проверкой памяти
    # ОПЦИИ VALGRIND:
    #   --leak-check=full - полная проверка утечек памяти
    #   --show-leak-kinds=all - показывать все виды утечек:
    #     definite - точные утечки
    #     indirect - косвенные утечки
    #     possible - возможные утечки
    #     reachable - достижимая но не освобождённая память
    #   --track-origins=yes - отслеживание источника неинициализированных значений
    #   --verbose - подробный вывод
    #   --error-exitcode=1 - код выхода при обнаружении ошибок
    #   --gen-suppressions=all - генерировать suppressions для ошибок
    #   --suppressions=file - файл с подавлениями ложных срабатываний
    # ДРУГИЕ ИНСТРУМЕНТЫ MEMCHECK:
    #   DrMemory (Windows) - альтернатива valgrind
    #   AddressSanitizer - встроенный в компилятор (быстрее valgrind)
    #   MemorySanitizer - для неинициализированной памяти
    # ВЕРСИЯ: CTest memcheck с CMake 2.8
    COMMAND ${CMAKE_CTEST_COMMAND} 
        --overwrite MemoryCheckCommand=/usr/bin/valgrind
        --overwrite MemoryCheckCommandOptions="--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose"
        -T memcheck
    DEPENDS ml_labs_tests
    COMMENT "Running tests with valgrind memory check..."
)

# Генерация отчёта о покрытии

# if - условный оператор.
# ML_LABS_ENABLE_COVERAGE - глобальная опция включения покрытия кода
if(ML_LABS_ENABLE_COVERAGE)
    add_custom_target(coverage_report
        # CMAKE_COMMAND - автоматическая переменная с путём к cmake
        # ОПРЕДЕЛЕНИЕ: Полный путь к исполняемому файлу cmake
        # -E - встроенные кроссплатформенные команды CMake
        # make_directory - создание директории (аналог mkdir -p)
        # СИНТАКСИС: cmake -E make_directory <dir>...
        # CMAKE_BINARY_DIR - корневая директория сборки
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        
        # lcov - инструмент для сбора данных покрытия от gcov
        # ОПЦИИ LCOV:
        #   --capture - режим захвата данных покрытия
        #   --directory dir - директория с .gcda/.gcno файлами
        #   --output-file file - выходной файл в формате .info
        #   --base-directory dir - базовая директория для относительных путей
        #   --initial - захват начального состояния (до запуска тестов)
        #   --no-external - исключить внешние файлы
        #   --quiet - минимальный вывод
        #   --rc setting=value - установка конфигурации
        # ФОРМАТ .info: текстовый формат с информацией о покрытии
        COMMAND lcov --capture --directory ${CMAKE_BINARY_DIR} --output-file coverage.info
        
        # lcov --remove - удаление ненужных файлов из отчёта
        # ПАТТЕРНЫ ИСКЛЮЧЕНИЯ:
        #   '/usr/*' - системные заголовки
        #   '*/tests/*' - сами тесты (нас интересует покрытие основного кода)
        #   '*/external/*' - внешние зависимости
        #   '*/build/*' - генерированные файлы
        #   '*.pb.cc' - protobuf генерированный код
        # СИНТАКСИС: lcov --remove input patterns... --output-file output
        COMMAND lcov --remove coverage.info '/usr/*' '*/tests/*' '*/external/*' --output-file coverage.info
        
        # lcov --list - вывод краткой статистики покрытия
        # ФОРМАТ ВЫВОДА:
        #   Filename                      Lines    Exec  Cover
        #   src/tensor.cpp                  100      85   85.0%
        #   src/autodiff.cpp                200     180   90.0%
        # ПОЛЕЗНО: Для быстрой оценки покрытия в консоли
        COMMAND lcov --list coverage.info
        
        # genhtml - генерация HTML отчёта из .info файла
        # ОПЦИИ GENHTML:
        #   coverage.info - входной файл с данными покрытия
        #   --output-directory dir - директория для HTML файлов
        #   --title "title" - заголовок отчёта
        #   --legend - показывать легенду
        #   --show-details - детальная информация по строкам
        #   --highlight - подсветка синтаксиса
        #   --ignore-errors source - игнорировать отсутствующие исходники
        #   --prefix dir - префикс для путей
        #   --num-spaces n - количество пробелов для табуляции
        #   --function-coverage - покрытие функций
        #   --branch-coverage - покрытие ветвлений
        # РЕЗУЛЬТАТ: index.html с навигацией по файлам и статистикой
        COMMAND genhtml coverage.info --output-directory ${CMAKE_BINARY_DIR}/coverage
        
        DEPENDS ml_labs_tests
        COMMENT "Generating coverage report..."
    )
    
    # message - вывод информационного сообщения
    # STATUS - уровень сообщения (префикс "-- ")
    message(STATUS "Coverage report will be in: ${CMAKE_BINARY_DIR}/coverage/index.html")
endif()

# Тесты производительности с baseline сравнением
add_custom_target(perf_regression
    # CMAKE_COMMAND -E env - установка переменных окружения для команды
    # СИНТАКСИС: cmake -E env [VAR=value]... COMMAND [args...]
    # ПЕРЕМЕННЫЕ:
    #   BENCHMARK_OUT - путь к выходному файлу (используется скриптом)
    # $<TARGET_FILE:target> - генераторное выражение для пути к файлу таргета
    # ОПРЕДЕЛЕНИЕ: Полный путь к исполняемому файлу созданному таргетом
    # GOOGLE BENCHMARK ОПЦИИ:
    #   --benchmark_out=file - выходной файл для результатов
    #   --benchmark_out_format=<console|json|csv> - формат вывода
    #   --benchmark_repetitions=n - количество повторений каждого теста
    #   --benchmark_report_aggregates_only=true - только агрегированные данные
    #   --benchmark_min_time=t - минимальное время для каждого benchmark
    #   --benchmark_filter=regex - фильтр тестов по имени
    #   --benchmark_list_tests - список всех тестов
    #   --benchmark_counters_tabular=true - табличный вывод счётчиков
    #   --benchmark_format=<console|json|csv> - формат консольного вывода
    #   --benchmark_context=key=value - контекст для результатов
    #   --benchmark_time_unit=<ns|us|ms|s> - единица времени
    # JSON ФОРМАТ: Структурированные данные для анализа и сравнения
    COMMAND ${CMAKE_COMMAND} -E env 
        BENCHMARK_OUT=${CMAKE_BINARY_DIR}/benchmark_results.json
        $<TARGET_FILE:ml_labs_bench_tests> 
        --benchmark_out=${CMAKE_BINARY_DIR}/benchmark_results.json
        --benchmark_out_format=json
        --benchmark_repetitions=10
        --benchmark_report_aggregates_only=true
    
    # Запуск Python скрипта для анализа регрессий
    # CMAKE_SOURCE_DIR - корневая директория исходников
    # СКРИПТ check_perf_regression.py:
    #   - Сравнивает текущие результаты с baseline
    #   - Выявляет регрессии производительности (>5% замедление)
    #   - Возвращает ненулевой код при обнаружении регрессий
    # АРГУМЕНТЫ:
    #   benchmark_results.json - текущие результаты
    #   baseline.json - эталонные результаты
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_perf_regression.py
        ${CMAKE_BINARY_DIR}/benchmark_results.json
        ${CMAKE_SOURCE_DIR}/benchmarks/baseline.json
    
    DEPENDS ml_labs_bench_tests
    COMMENT "Running performance regression tests..."
)

# Установка таймаутов для разных категорий тестов

# set_tests_properties - установка свойств для тестов CTest.
# СИГНАТУРА: set_tests_properties(test1 [test2...] PROPERTIES prop1 value1 
#                                  [prop2 value2...])
# ПАРАМЕТРЫ:
#   test - имена тестов (могут содержать wildcards)
#   PROPERTIES - ключевое слово начала списка свойств
#   prop value - пары свойство-значение
# СВОЙСТВА ТЕСТОВ:
#   TIMEOUT - максимальное время выполнения в секундах
#   COST - относительная стоимость для планировщика параллельного запуска
#   DEPENDS - зависимости от других тестов
#   DISABLED - тест отключен
#   ENVIRONMENT - переменные окружения (VAR=value;VAR2=value2)
#   FAIL_REGULAR_EXPRESSION - regex указывающий на провал
#   PASS_REGULAR_EXPRESSION - regex указывающий на успех
#   SKIP_REGULAR_EXPRESSION - regex для пропуска теста
#   LABELS - метки для группировки тестов
#   MEASUREMENT - измерения для CDash
#   PROCESSORS - количество процессоров/потоков
#   PROCESSOR_AFFINITY - привязка к процессорам
#   REQUIRED_FILES - необходимые файлы
#   RESOURCE_LOCK - именованная блокировка ресурса
#   RUN_SERIAL - запускать последовательно (не параллельно)
#   SKIP_RETURN_CODE - код возврата означающий пропуск
#   WILL_FAIL - тест должен провалиться для успеха
#   WORKING_DIRECTORY - рабочая директория
#   ATTACHED_FILES - файлы для CDash
#   ATTACHED_FILES_ON_FAIL - файлы при провале
# ПЕРЕМЕННЫЕ СО СПИСКАМИ ТЕСТОВ:
#   UNIT_TEST_LIST - должна быть заполнена где-то выше
#   INTEGRATION_TEST_LIST - должна быть заполнена где-то выше
# ПРИМЕЧАНИЕ: Эти переменные обычно заполняются через gtest_discover_tests
#             или явно через get_property(... TEST ...)
# ВЕРСИЯ: Доступна с CMake 2.8
set_tests_properties(${UNIT_TEST_LIST} PROPERTIES TIMEOUT 10)      # 10 секунд для unit
set_tests_properties(${INTEGRATION_TEST_LIST} PROPERTIES TIMEOUT 60) # 1 минута для integration

# Дополнительные свойства для категорий тестов
if(DEFINED UNIT_TEST_LIST)
    set_tests_properties(${UNIT_TEST_LIST} PROPERTIES 
        LABELS "unit;fast"
        RUN_SERIAL FALSE  # Могут выполняться параллельно
    )
endif()

if(DEFINED INTEGRATION_TEST_LIST)
    set_tests_properties(${INTEGRATION_TEST_LIST} PROPERTIES
        LABELS "integration;slow"
        RUN_SERIAL TRUE   # Выполнять последовательно
        PROCESSORS 2      # Требуют больше ресурсов
    )
endif()
