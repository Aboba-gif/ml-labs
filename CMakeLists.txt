# Корневой файл системы сборки для ML Labs - высокопроизводительной библиотеки машинного обучения.
# Copyright (C) 2024 ML Labs Project
#
# АРХИТЕКТУРА СИСТЕМЫ СБОРКИ:
# ===========================
# Используется Modern CMake (3.25+) с target-centric подходом.
# Все зависимости явные через target_link_libraries.
# Никаких глобальных переменных или include_directories.
#
# ИЕРАРХИЯ ТАРГЕТОВ:
# ml_labs::core       - основная библиотека (STATIC)
# ml_labs::tests      - модульные тесты (EXECUTABLE)  
# ml_labs::benchmarks - бенчмарки производительности (EXECUTABLE)
# ml_labs::lab_*      - индивидуальные лабораторные (EXECUTABLE)
#
# ПРОИЗВОДИТЕЛЬНОСТЬ СБОРКИ:
# - Precompiled headers для ускорения на 30-40%
# - Unity builds опционально для CI (-DCMAKE_UNITY_BUILD=ON)
# - ccache автоматически если установлен
# - Parallel builds: -j$(nproc)
#
# ССЫЛКИ:
# [1] "Effective Modern CMake" - https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1
# [2] "Professional CMake" - Craig Scott, 2018

# cmake_minimum_required - устанавливает минимальную версию CMake для проекта.
# СИГНАТУРА: cmake_minimum_required(VERSION <min>[...<policy_max>] [FATAL_ERROR])
# ПАРАМЕТРЫ:
#   VERSION - ключевое слово, обозначающее начало версии
#   <min> - минимальная требуемая версия (формат major[.minor[.patch[.tweak]]])
#   ...<policy_max> - опциональная максимальная версия для политик (CMake 3.12+)
#   FATAL_ERROR - прерывает выполнение с ошибкой если версия ниже (legacy для CMake < 2.6)
# ПОБОЧНЫЕ ЭФФЕКТЫ:
#   - Устанавливает CMAKE_MINIMUM_REQUIRED_VERSION
#   - Устанавливает политики (policies) CMake до версии <min>
#   - Вызывает cmake_policy(VERSION <min>)
# ДЕТАЛИ: Политики контролируют поведение CMake для обратной совместимости.
#         Каждая версия CMake может изменять поведение, политики позволяют
#         контролировать эти изменения.
# ВЕРСИЯ: Доступна с CMake 2.6, расширенный синтаксис с 3.12
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Защита от загрязнения build дерева исходниками.
# ОБОСНОВАНИЕ: In-source builds создают .o файлы рядом с .cpp,
# делая невозможным clean build и загрязняя git status.

# if() - условный оператор для ветвления логики.
# СИГНАТУРА: if(<condition>) ... elseif(<condition>) ... else() ... endif()
# УСЛОВИЯ:
#   <variable> - проверка что переменная TRUE (не 0, OFF, FALSE, N, NO, "", NOTFOUND, -NOTFOUND)
#   NOT <condition> - логическое отрицание
#   <variable> AND <variable> - логическое И
#   <variable> OR <variable> - логическое ИЛИ
#   <variable> STREQUAL <string> - лексическое сравнение строк
#   <variable> MATCHES <regex> - сопоставление с регулярным выражением
#   EXISTS <path> - проверка существования файла/директории
#   IS_DIRECTORY <path> - проверка что путь является директорией
#   DEFINED <variable> - проверка что переменная определена
# CMAKE_SOURCE_DIR - автоматическая переменная с абсолютным путём к корню исходников
# CMAKE_BINARY_DIR - автоматическая переменная с абсолютным путём к корню build директории
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    # message() - вывод сообщений в консоль и логи.
    # СИГНАТУРА: message([<mode>] "message text" ...)
    # РЕЖИМЫ:
    #   (none) - важное сообщение (всегда печатается)
    #   STATUS - информационное сообщение с префиксом "-- "
    #   WARNING - предупреждение (жёлтый цвет в терминале)
    #   AUTHOR_WARNING - предупреждение для разработчиков CMakeLists.txt
    #   SEND_ERROR - ошибка, продолжает конфигурацию но пропускает генерацию
    #   FATAL_ERROR - критическая ошибка, немедленная остановка с кодом 1
    #   DEPRECATION - сообщение об устаревании (контролируется CMAKE_ERROR_DEPRECATED)
    # ДЕТАЛИ: Сообщения конкатенируются без пробелов между аргументами
    # ВЕРСИЯ: Базовая с CMake 2.6, режимы добавлялись постепенно
    message(FATAL_ERROR 
        "In-source builds are not allowed.\n"
        "Please create a build directory and run cmake from there:\n"
        "  mkdir build && cd build && cmake ..\n"
        "To remove CMakeCache.txt and CMakeFiles/ from source directory:\n"
        "  rm -rf CMakeCache.txt CMakeFiles/")
# endif() - обязательное завершение блока if
# СЕМАНТИКА: Каждый if() должен иметь соответствующий endif()
endif()

# Метаданные проекта.
# VERSION используется для SONAME в shared libraries и package managers.
# LANGUAGES CXX означает что C компилятор не нужен (ускоряет configure).

# project() - объявляет проект и устанавливает его метаданные.
# СИГНАТУРА: project(<PROJECT-NAME> [<language-name>...])
#            project(<PROJECT-NAME>
#                    [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
#                    [DESCRIPTION <project-description-string>]
#                    [HOMEPAGE_URL <url-string>]
#                    [LANGUAGES <language-name>...])
# ПАРАМЕТРЫ:
#   PROJECT-NAME - имя проекта (буквы, цифры, подчёркивания, дефисы)
#   VERSION - версия в формате major.minor.patch.tweak
#   DESCRIPTION - краткое описание проекта (для pkg-config и прочих)
#   HOMEPAGE_URL - URL домашней страницы проекта
#   LANGUAGES - список языков программирования (C, CXX, Fortran, ASM, CUDA, etc.)
# ПОБОЧНЫЕ ЭФФЕКТЫ:
#   - Устанавливает PROJECT_NAME, PROJECT_VERSION, PROJECT_DESCRIPTION и др.
#   - Создаёт PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR
#   - Устанавливает <PROJECT-NAME>_VERSION_MAJOR/MINOR/PATCH/TWEAK
#   - Вызывает enable_language() для каждого языка
#   - Детектирует и инициализирует компиляторы
# ВЕРСИЯ: Базовая с CMake 2.6, расширенный синтаксис с 3.0
project(ml_labs
    VERSION 0.1.0
    DESCRIPTION "High-performance machine learning laboratory framework in modern C++"
    HOMEPAGE_URL "https://github.com/yourusername/ml-labs"
    LANGUAGES CXX
)

# Установка стандарта C++.
# C++23 для std::expected, std::mdspan, std::print и других современных фич.
# ОБОСНОВАНИЕ выбора C++23:
# - std::expected для error handling без исключений
# - std::mdspan для многомерных массивов без overhead
# - std::stacktrace для debugging
# - Deducing this для CRTP оптимизаций

# set() - установка значения переменной в текущей области видимости.
# СИГНАТУРА: set(<variable> <value>... [PARENT_SCOPE])
# ПАРАМЕТРЫ:
#   variable - имя переменной (регистрозависимое)
#   value - одно или несколько значений (создают список через ;)
#   PARENT_SCOPE - установить в родительской области видимости
# СЕМАНТИКА: Создаёт или перезаписывает переменную в текущем scope
# CMAKE_CXX_STANDARD - специальная переменная для стандарта ISO C++
# ДОПУСТИМЫЕ ЗНАЧЕНИЯ: 98, 11, 14, 17, 20, 23, 26 (зависит от версии CMake)
# ЭФФЕКТ: Добавляет -std=c++23 для GCC/Clang или /std:c++23 для MSVC
set(CMAKE_CXX_STANDARD 23)
# CMAKE_CXX_STANDARD_REQUIRED - требовать точный стандарт или выше
# ON - не откатываться на старые стандарты если запрошенный недоступен
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CMAKE_CXX_EXTENSIONS - использование расширений компилятора
# OFF - использовать -std=c++23 вместо -std=gnu++23
set(CMAKE_CXX_EXTENSIONS OFF)

# Экспорт compile_commands.json для clangd/IDE.
# Файл содержит точные команды компиляции для каждого .cpp файла.
# Используется для автодополнения, рефакторинга, статического анализа.
# CMAKE_EXPORT_COMPILE_COMMANDS - генерация compile_commands.json
# ФОРМАТ: JSON array с объектами {"directory", "command", "file"}
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Включение Position Independent Code для shared libraries.
# На x86_64 добавляет ~2% overhead, но необходимо для .so
# CMAKE_POSITION_INDEPENDENT_CODE - глобальное включение PIC
# ЭФФЕКТ: Добавляет -fPIC для GCC/Clang, игнорируется на Windows
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Установка RPATH для правильной работы с shared libraries.
# BUILD_RPATH для запуска из build директории
# INSTALL_RPATH для установленных бинарников
# CMAKE_BUILD_RPATH_USE_ORIGIN - использовать $ORIGIN в RPATH
# $ORIGIN - специальная переменная линкера означающая директорию исполняемого файла
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# Включение inter-procedural optimization (LTO) в Release.
# Увеличивает время линковки на 50-100%, но даёт 5-15% performance gain.
# ОБОСНОВАНИЕ: ML код часто имеет горячие циклы которые выигрывают от инлайнинга.

# include() - подключение CMake модуля или файла.
# СИГНАТУРА: include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>] [NO_POLICY_SCOPE])
# ПАРАМЕТРЫ:
#   file|module - путь к файлу или имя модуля
#   OPTIONAL - не выдавать ошибку если файл не найден
#   RESULT_VARIABLE - переменная для полного пути найденного файла
#   NO_POLICY_SCOPE - не создавать новую область для политик
# ПОИСК МОДУЛЕЙ: CMAKE_MODULE_PATH, затем <CMAKE_ROOT>/Modules
# CheckIPOSupported - встроенный модуль проверки поддержки IPO/LTO
include(CheckIPOSupported)

# check_ipo_supported() - проверяет поддержку Inter-Procedural Optimization.
# СИГНАТУРА: check_ipo_supported([RESULT <result>] [OUTPUT <output>] 
#                                 [LANGUAGES <lang>...])
# ПАРАМЕТРЫ:
#   RESULT - переменная для булевого результата
#   OUTPUT - переменная для сообщения об ошибке
#   LANGUAGES - список языков для проверки (по умолчанию C и CXX)
# МЕХАНИЗМ: Компилирует тестовую программу с IPO флагами
# ВЕРСИЯ: Доступна с CMake 3.9
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    # CMAKE_INTERPROCEDURAL_OPTIMIZATION_<CONFIG> - включение IPO для конфигурации
    # ЭФФЕКТ: Добавляет -flto для GCC/Clang, /GL для MSVC
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    message(STATUS "IPO/LTO enabled for Release builds")
else()
    message(STATUS "IPO/LTO not supported: ${IPO_ERROR}")
endif()

# Поиск ccache для ускорения повторных компиляций.
# ccache кеширует объектные файлы по хешу препроцессированного кода.
# Ускорение повторных сборок: 5-10x

# find_program() - поиск исполняемого файла в системных путях.
# СИГНАТУРА: find_program(<VAR> name1 [path1 path2 ...]
#                         [NAMES name1 [name2 ...]]
#                         [HINTS [path | ENV var]...]
#                         [PATHS [path | ENV var]...]
#                         [PATH_SUFFIXES suffix1 [suffix2 ...]]
#                         [DOC "cache documentation string"]
#                         [REQUIRED]
#                         [NO_DEFAULT_PATH]
#                         [NO_PACKAGE_ROOT_PATH]
#                         [NO_CMAKE_PATH]
#                         [NO_CMAKE_ENVIRONMENT_PATH]
#                         [NO_SYSTEM_ENVIRONMENT_PATH]
#                         [NO_CMAKE_SYSTEM_PATH]
#                         [CMAKE_FIND_ROOT_PATH_BOTH |
#                          ONLY_CMAKE_FIND_ROOT_PATH |
#                          NO_CMAKE_FIND_ROOT_PATH])
# ВОЗВРАЩАЕТ: Полный путь к программе или <VAR>-NOTFOUND
# ПОИСК: CMAKE_PREFIX_PATH, PATH, стандартные системные директории
# КЭШИРОВАНИЕ: Результат сохраняется в CMakeCache.txt
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # CMAKE_CXX_COMPILER_LAUNCHER - программа-обёртка для компилятора
    # ЭФФЕКТ: Все вызовы компилятора проходят через launcher
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    
    # Настройка ccache для оптимальной производительности
    # set(ENV{VAR}) - установка переменной окружения
    # СИНТАКСИС: set(ENV{variable} value)
    # ОБЛАСТЬ: Только для текущего процесса CMake и его потомков
    set(ENV{CCACHE_COMPRESS} "1")          # Сжатие кеша (экономия места)
    set(ENV{CCACHE_COMPRESSLEVEL} "6")     # Уровень zstd компрессии
    set(ENV{CCACHE_MAXSIZE} "10G")         # Максимальный размер кеша
    set(ENV{CCACHE_HARDLINK} "1")          # Hardlinks вместо копирования
endif()

# Опции проекта с описаниями для cmake-gui/ccmake.
# CACHE BOOL создаёт персистентные переменные в CMakeCache.txt

# option() - объявляет булевую опцию с описанием.
# СИГНАТУРА: option(<variable> "<help_text>" [value])
# ПАРАМЕТРЫ:
#   variable - имя переменной опции
#   help_text - описание для --help и GUI
#   value - начальное значение (ON/OFF, по умолчанию OFF)
# ДЕТАЛИ: Создаёт CACHE переменную типа BOOL
# ИСПОЛЬЗОВАНИЕ: cmake -DML_LABS_BUILD_TESTS=OFF ..
option(ML_LABS_BUILD_TESTS 
    "Build the unit tests suite" 
    ON)
    
option(ML_LABS_BUILD_BENCHMARKS 
    "Build performance benchmarks" 
    ON)
    
option(ML_LABS_BUILD_DOCS 
    "Build API documentation with Doxygen" 
    ${DOXYGEN_FOUND})
    
option(ML_LABS_BUILD_EXAMPLES 
    "Build example applications" 
    ON)
    
option(ML_LABS_ENABLE_COVERAGE 
    "Enable code coverage instrumentation (Debug only)" 
    OFF)
    
option(ML_LABS_ENABLE_SANITIZERS 
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer (Debug only)" 
    OFF)
    
option(ML_LABS_ENABLE_CLANG_TIDY 
    "Run clang-tidy static analysis during build" 
    OFF)
    
option(ML_LABS_ENABLE_CPPCHECK 
    "Run cppcheck static analysis during build" 
    OFF)
    
option(ML_LABS_USE_VCPKG 
    "Use vcpkg for dependency management instead of Conan" 
    OFF)

# Валидация опций.
# Sanitizers и Coverage только в Debug из-за overhead.
# CMAKE_BUILD_TYPE - тип конфигурации сборки
# ЗНАЧЕНИЯ: Debug, Release, RelWithDebInfo, MinSizeRel, или пользовательские
if(ML_LABS_ENABLE_COVERAGE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Coverage enabled only in Debug builds, ignoring")
    set(ML_LABS_ENABLE_COVERAGE OFF)
endif()

if(ML_LABS_ENABLE_SANITIZERS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Sanitizers enabled only in Debug builds, ignoring")
    set(ML_LABS_ENABLE_SANITIZERS OFF)
endif()

# Подключение модулей CMake.
# CMAKE_MODULE_PATH расширяет пути поиска для include()

# list() - операции со списками (массивами строк).
# СИГНАТУРА: list(<operation> <list> [<element>...] [OUTPUT_VARIABLE <var>])
# ОПЕРАЦИИ:
#   APPEND <list> <element>... - добавить в конец
#   PREPEND <list> <element>... - добавить в начало (CMake 3.15+)
#   INSERT <list> <index> <element>... - вставить по индексу
#   REMOVE_ITEM <list> <value>... - удалить по значению
#   REMOVE_AT <list> <index>... - удалить по индексу
#   REMOVE_DUPLICATES <list> - удалить дубликаты
#   REVERSE <list> - развернуть список
#   SORT <list> - отсортировать
#   LENGTH <list> <var> - получить длину
#   GET <list> <index> <var> - получить элемент
#   FIND <list> <value> <var> - найти индекс элемента
# CMAKE_CURRENT_SOURCE_DIR - путь к директории текущего CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Подключение наших модулей
include(CompilerWarnings)   # Настройка предупреждений компилятора
include(Sanitizers)         # Address/Thread/UB sanitizers
include(Coverage)           # Code coverage с gcov/lcov

# Статический анализ с clang-tidy.
# clang-tidy проверяет: modernize-*, performance-*, bugprone-*, readability-*
if(ML_LABS_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXECUTABLE clang-tidy)
    if(CLANG_TIDY_EXECUTABLE)
        # CMAKE_CXX_CLANG_TIDY - программа статического анализа для C++
        # ФОРМАТ: программа [аргументы]
        # ЭФФЕКТ: Запускается для каждого компилируемого файла
        set(CMAKE_CXX_CLANG_TIDY 
            ${CLANG_TIDY_EXECUTABLE}
            --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
            --header-filter=${CMAKE_SOURCE_DIR}/include/.*
        )
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXECUTABLE}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endif()

# Настройка Conan для управления зависимостями.
# Conan 2.x использует новый CMakeDeps generator.
if(NOT ML_LABS_USE_VCPKG)
    # EXISTS - проверка существования файла или директории
    # СИНТАКСИС: if(EXISTS path)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        
        # file() - операции с файловой системой.
        # СИГНАТУРА: file(DOWNLOAD <url> <file> [options...])
        # ОПЕРАЦИИ:
        #   DOWNLOAD url file - загрузить файл
        #   UPLOAD file url - загрузить на сервер
        #   READ file variable - прочитать в переменную
        #   WRITE file content - записать в файл
        #   APPEND file content - дописать в файл
        #   GLOB variable pattern - найти файлы по паттерну
        #   COPY files... DESTINATION - копировать файлы
        #   INSTALL files... DESTINATION - установить файлы
        #   REMOVE files... - удалить файлы
        #   MAKE_DIRECTORY dirs... - создать директории
        # ОПЦИИ DOWNLOAD:
        #   INACTIVITY_TIMEOUT seconds - таймаут неактивности
        #   TIMEOUT seconds - общий таймаут
        #   STATUS variable - статус загрузки [код;сообщение]
        #   EXPECTED_HASH algo=hash - проверка контрольной суммы
        #   TLS_VERIFY on|off - проверка SSL сертификата
        #   TLS_CAINFO file - файл CA сертификатов
        #   SHOW_PROGRESS - показывать прогресс
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/conan-io/cmake-conan/develop2/conan_provider.cmake"
            "${CMAKE_BINARY_DIR}/conan_provider.cmake"
            TLS_VERIFY ON
            STATUS DOWNLOAD_STATUS
        )
        
        # list(GET) - получение элемента списка.
        # СИГНАТУРА: list(GET <list> <element index>... <output variable>)
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
        # EQUAL - оператор сравнения чисел
        if(NOT DOWNLOAD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to download conan.cmake")
        endif()
    endif()
    
    # CMAKE_PROJECT_TOP_LEVEL_INCLUDES - файлы включаемые в начале проекта (CMake 3.24+)
    set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES ${CMAKE_BINARY_DIR}/conan_provider.cmake)
    
    # execute_process() - запуск внешней команды во время конфигурации.
    # СИГНАТУРА: execute_process(COMMAND <cmd1> [<arguments>]
    #                            [COMMAND <cmd2> [<arguments>]]...
    #                            [WORKING_DIRECTORY <directory>]
    #                            [TIMEOUT <seconds>]
    #                            [RESULT_VARIABLE <variable>]
    #                            [RESULTS_VARIABLE <variable>]
    #                            [OUTPUT_VARIABLE <variable>]
    #                            [ERROR_VARIABLE <variable>]
    #                            [INPUT_FILE <file>]
    #                            [OUTPUT_FILE <file>]
    #                            [ERROR_FILE <file>]
    #                            [OUTPUT_QUIET]
    #                            [ERROR_QUIET]
    #                            [COMMAND_ECHO <where>]
    #                            [OUTPUT_STRIP_TRAILING_WHITESPACE]
    #                            [ERROR_STRIP_TRAILING_WHITESPACE]
    #                            [ENCODING <name>]
    #                            [ECHO_OUTPUT_VARIABLE]
    #                            [ECHO_ERROR_VARIABLE]
    #                            [COMMAND_ERROR_IS_FATAL <ANY|LAST>])
    # ДЕТАЛИ: COMMAND можно chain через pipe
    # ВЕРСИЯ: Базовая с CMake 2.6, расширения в 3.15+
    execute_process(
        COMMAND conan install ${CMAKE_SOURCE_DIR} 
                --output-folder=${CMAKE_BINARY_DIR}
                --build=missing
                --settings=build_type=${CMAKE_BUILD_TYPE}
        RESULT_VARIABLE CONAN_RESULT
    )
    if(NOT CONAN_RESULT EQUAL 0)
        message(FATAL_ERROR "Conan install failed")
    endif()
endif()

# Поиск системных библиотек.
# Threads необходим для std::thread, std::mutex

# find_package() - поиск и загрузка настроек внешнего пакета.
# СИГНАТУРА: find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
#                         [REQUIRED] [[COMPONENTS] [components...]]
#                         [OPTIONAL_COMPONENTS components...]
#                         [CONFIG|NO_MODULE]
#                         [NO_POLICY_SCOPE]
#                         [NAMES name1 [name2 ...]]
#                         [CONFIGS config1 [config2 ...]]
#                         [HINTS path1 [path2 ... ]]
#                         [PATHS path1 [path2 ... ]]
#                         [PATH_SUFFIXES suffix1 [suffix2 ...]]
#                         [NO_DEFAULT_PATH]
#                         [NO_PACKAGE_ROOT_PATH]
#                         [NO_CMAKE_PATH]
#                         [NO_CMAKE_ENVIRONMENT_PATH]
#                         [NO_SYSTEM_ENVIRONMENT_PATH]
#                         [NO_CMAKE_BUILDS_PATH]
#                         [NO_CMAKE_PACKAGE_REGISTRY]
#                         [NO_CMAKE_SYSTEM_PATH]
#                         [NO_CMAKE_SYSTEM_PACKAGE_REGISTRY]
#                         [CMAKE_FIND_ROOT_PATH_BOTH |
#                          ONLY_CMAKE_FIND_ROOT_PATH |
#                          NO_CMAKE_FIND_ROOT_PATH]
#                         [CMAKE_FIND_PACKAGE_SORT_ORDER <order>]
#                         [CMAKE_FIND_PACKAGE_SORT_DIRECTION <direction>]
#                         [REGISTRY_VIEW <64|32|64_32|32_64|HOST|TARGET|BOTH>]
#                         [BYPASS_PROVIDER]
#                         [GLOBAL])
# РЕЖИМЫ:
#   Module mode - ищет Find<PackageName>.cmake
#   Config mode - ищет <PackageName>Config.cmake
# ВОЗВРАЩАЕТ:
#   <PackageName>_FOUND - найден ли пакет
#   <PackageName>_VERSION - версия пакета
#   <PackageName>_LIBRARIES - библиотеки для линковки
#   <PackageName>_INCLUDE_DIRS - директории заголовков
# REQUIRED - прерывание с ошибкой если не найден
find_package(Threads REQUIRED)

# OpenMP для параллельных вычислений в ML алгоритмах
find_package(OpenMP)
# <Package>_<Lang>_FOUND - стандартная переменная для языко-специфичных пакетов
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP not found, parallel algorithms will use std::execution")
endif()

# Основная библиотека проекта.
# STATIC по умолчанию для лучшей оптимизации и инлайнинга.
# Можно переопределить через BUILD_SHARED_LIBS=ON

# add_library() - создание библиотечного таргета.
# СИГНАТУРА: add_library(<name> [STATIC | SHARED | MODULE | INTERFACE | OBJECT | ALIAS]
#                        [EXCLUDE_FROM_ALL] [<source>...])
# ТИПЫ:
#   STATIC - статическая библиотека (.a/.lib)
#   SHARED - динамическая библиотека (.so/.dll/.dylib)
#   MODULE - загружаемый модуль (dlopen/LoadLibrary)
#   INTERFACE - header-only библиотека
#   OBJECT - объектные файлы для переиспользования
#   ALIAS - псевдоним существующей библиотеки
# EXCLUDE_FROM_ALL - не собирать по умолчанию (только если затребована)
# ВЕРСИЯ: Базовая с CMake 2.6, INTERFACE с 3.0, ALIAS с 2.8.11
add_library(ml_labs_core STATIC)

# Источники основной библиотеки.
# PRIVATE источники не влияют на интерфейс

# target_sources() - добавление исходных файлов к таргету.
# СИГНАТУРА: target_sources(<target>
#                           <INTERFACE|PUBLIC|PRIVATE> [items1...]
#                           [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...]
#                           [FILE_SET <set> [TYPE <type>] [BASE_DIRS <dirs>...] FILES <files>...])
# ПАРАМЕТРЫ:
#   target - имя цели к которой добавляются файлы
#   INTERFACE - файлы только для потребителей таргета
#   PUBLIC - файлы для таргета и его потребителей
#   PRIVATE - файлы только для компиляции таргета
#   FILE_SET - именованный набор файлов (CMake 3.23+)
#   TYPE - тип набора (HEADERS, CXX_MODULES)
#   BASE_DIRS - базовые директории для относительных путей
# ВЕРСИЯ: Базовая с CMake 3.1, FILE_SET с 3.23
target_sources(ml_labs_core PRIVATE
    src/core/tensor.cpp
    src/core/autodiff.cpp
    src/core/optimizer.cpp
    src/core/loss.cpp
    src/core/activation.cpp
)

# Публичные заголовки.
# PUBLIC headers устанавливаются и экспортируются
# FILE_SET HEADERS - специальный набор для заголовочных файлов
# BASE_DIRS - базовые пути для вычисления относительных путей при установке
target_sources(ml_labs_core PUBLIC 
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/ml_labs/core/tensor.hpp
        include/ml_labs/core/autodiff.hpp
        include/ml_labs/core/optimizer.hpp
        include/ml_labs/core/loss.hpp
        include/ml_labs/core/activation.hpp
        include/ml_labs/core/concepts.hpp
)

# Include пути.
# BUILD_INTERFACE активен при сборке
# INSTALL_INTERFACE активен после установки

# target_include_directories() - установка путей поиска заголовков для таргета.
# СИГНАТУРА: target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
#                                        <INTERFACE|PUBLIC|PRIVATE> [items1...]
#                                        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
# ПАРАМЕТРЫ:
#   target - имя цели
#   SYSTEM - пометить как системные (подавляет предупреждения)
#   AFTER|BEFORE - добавить в конец или начало списка
#   INTERFACE|PUBLIC|PRIVATE - область видимости
# ГЕНЕРАТОРНЫЕ ВЫРАЖЕНИЯ:
#   $<BUILD_INTERFACE:...> - активно только при сборке
#   $<INSTALL_INTERFACE:...> - активно только после установки
#   $<BOOL:condition> - условное включение
#   $<$<CONFIG:Debug>:...> - только для Debug конфигурации
# ВЕРСИЯ: Доступна с CMake 2.8.11
target_include_directories(ml_labs_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Зависимости через Conan.
# Conan генерирует Find*.cmake файлы
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

# target_link_libraries() - связывание таргета с библиотеками.
# СИГНАТУРА: target_link_libraries(<target>
#                                   <PRIVATE|PUBLIC|INTERFACE> <item>...
#                                   [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
# ПАРАМЕТРЫ:
#   target - имя цели для линковки
#   PRIVATE - зависимости только для реализации
#   PUBLIC - зависимости для реализации и интерфейса
#   INTERFACE - зависимости только для интерфейса
#   item - библиотека, таргет или генераторное выражение
# ГЕНЕРАТОРНЫЕ ВЫРАЖЕНИЯ:
#   $<$<BOOL:${var}>:item> - условная линковка
#   $<$<CONFIG:Debug>:debug_lib> - линковка только в Debug
#   $<$<CXX_COMPILER_ID:GNU>:gcc_lib> - линковка для конкретного компилятора
# ВЕРСИЯ: Базовая с CMake 2.6, INTERFACE/PUBLIC/PRIVATE с 2.8.11
target_link_libraries(ml_labs_core
    PUBLIC
        Eigen3::Eigen      # Линейная алгебра
        Threads::Threads   # Многопоточность
    PRIVATE  
        fmt::fmt          # Форматирование
        spdlog::spdlog    # Логирование
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Применение настроек предупреждений
# set_project_warnings - пользовательская функция из CompilerWarnings.cmake
# СИГНАТУРА: set_project_warnings(target_name [WARNING_LEVEL level])
# ПАРАМЕТРЫ:
#   target_name - имя таргета
#   WARNING_LEVEL - INTERFACE|PUBLIC|PRIVATE (по умолчанию PRIVATE)
# ЭФФЕКТ: Устанавливает максимально строгие предупреждения для таргета
set_project_warnings(ml_labs_core)

# Включение sanitizers если запрошено
if(ML_LABS_ENABLE_SANITIZERS)
    # enable_sanitizers - пользовательская функция из Sanitizers.cmake
    # СИГНАТУРА: enable_sanitizers(target SANITIZERS sanitizer1 [sanitizer2 ...])
    # ПАРАМЕТРЫ:
    #   target - имя таргета
    #   SANITIZERS - ключевое слово перед списком
    #   sanitizer - address|thread|undefined|memory|leak
    # ЭФФЕКТ: Добавляет флаги компиляции и линковки для sanitizers
    # ДЕТАЛИ: TSan несовместим с ASan, MSan несовместим со всеми
    enable_sanitizers(ml_labs_core SANITIZERS address undefined)
endif()

# Включение code coverage
if(ML_LABS_ENABLE_COVERAGE)
    # enable_coverage - пользовательская функция из Coverage.cmake
    # СИГНАТУРА: enable_coverage(target [EXCLUDE pattern1 ...])
    # ПАРАМЕТРЫ:
    #   target - имя таргета
    #   EXCLUDE - паттерны файлов для исключения из покрытия
    # ЭФФЕКТ: Добавляет --coverage и другие флаги для генерации .gcda/.gcno
    enable_coverage(ml_labs_core)
endif()

# Precompiled headers для ускорения компиляции.
# PCH кешируют парсинг часто используемых заголовков.
# Ускорение: 30-40% для проектов с STL/Boost

# target_precompile_headers() - установка предкомпилированных заголовков.
# СИГНАТУРА: target_precompile_headers(<target>
#                                       <INTERFACE|PUBLIC|PRIVATE> [header1...]
#                                       [<INTERFACE|PUBLIC|PRIVATE> [header2...] ...]
#                                       [REUSE_FROM <other_target>])
# ПАРАМЕТРЫ:
#   target - имя цели
#   INTERFACE|PUBLIC|PRIVATE - область видимости
#   header - заголовок для предкомпиляции (<header> или "header.h")
#   REUSE_FROM - переиспользовать PCH от другого таргета
# ФОРМАТ ЗАГОЛОВКОВ:
#   <header> - системный заголовок
#   "header.h" - пользовательский заголовок
#   "path/to/header.h" - заголовок с путём
# ДЕТАЛИ: Создаёт .pch (MSVC) или .gch (GCC/Clang) файлы
# ВЕРСИЯ: Доступна с CMake 3.16
target_precompile_headers(ml_labs_core PRIVATE
    <algorithm>
    <array>
    <atomic>
    <concepts>
    <expected>
    <memory>
    <span>
    <vector>
    <Eigen/Core>
)

# Создание alias для удобства использования

# add_library() с ALIAS - создание псевдонима для существующего таргета.
# СИГНАТУРА: add_library(<name> ALIAS <target>)
# ПАРАМЕТРЫ:
#   name - имя псевдонима (обычно с namespace::)
#   ALIAS - ключевое слово
#   target - существующий таргет
# ИСПОЛЬЗОВАНИЕ: Позволяет использовать namespace::name синтаксис
# ДЕТАЛИ: Псевдоним не может быть экспортирован или установлен
# ВЕРСИЯ: Доступна с CMake 2.8.11
add_library(ml_labs::core ALIAS ml_labs_core)

# Подключение поддиректорий

# add_subdirectory() - добавление поддиректории для обработки.
# СИГНАТУРА: add_subdirectory(source_dir [binary_dir] 
#                             [EXCLUDE_FROM_ALL] [SYSTEM])
# ПАРАМЕТРЫ:
#   source_dir - путь к поддиректории с CMakeLists.txt
#   binary_dir - путь для build файлов (по умолчанию source_dir)
#   EXCLUDE_FROM_ALL - не включать таргеты в ALL
#   SYSTEM - пометить include пути как системные (CMake 3.25+)
# ЭФФЕКТ: Обрабатывает CMakeLists.txt в поддиректории
# ОБЛАСТЬ ВИДИМОСТИ: Создаёт новую область для переменных
# ВЕРСИЯ: Базовая с CMake 2.6, SYSTEM с 3.25
add_subdirectory(labs)

if(ML_LABS_BUILD_TESTS)
    # enable_testing() - включение инфраструктуры тестирования CTest.
    # СИГНАТУРА: enable_testing()
    # ЭФФЕКТ:
    #   - Создаёт таргет 'test' (или RUN_TESTS в VS)
    #   - Включает команды add_test()
    #   - Генерирует CTestTestfile.cmake
    #   - Позволяет запускать тесты через ctest
    # ОБЛАСТЬ: Должна быть вызвана в корневом CMakeLists.txt
    # ВЕРСИЯ: Доступна с CMake 2.6
    enable_testing()
    
    # include(CTest) - расширенная настройка CTest
    # ЭФФЕКТ:
    #   - Включает enable_testing()
    #   - Добавляет опции BUILD_TESTING
    #   - Настраивает submission в CDash dashboard
    #   - Создаёт CTestConfig.cmake
    # ПЕРЕМЕННЫЕ:
    #   BUILD_TESTING - основная опция включения тестов
    #   DART_TESTING_TIMEOUT - таймаут для тестов
    #   MEMORYCHECK_COMMAND - команда для valgrind
    include(CTest)
    add_subdirectory(tests)
endif()

if(ML_LABS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ML_LABS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Установка и экспорт.
# GNUInstallDirs обеспечивает правильные пути для Linux/BSD

# include(GNUInstallDirs) - стандартные переменные путей установки GNU.
# ПРЕДОСТАВЛЯЕТ:
#   CMAKE_INSTALL_PREFIX - корень установки (/usr/local по умолчанию)
#   CMAKE_INSTALL_BINDIR - исполняемые файлы (bin)
#   CMAKE_INSTALL_SBINDIR - системные исполняемые (sbin)
#   CMAKE_INSTALL_LIBEXECDIR - вспомогательные программы (libexec)
#   CMAKE_INSTALL_SYSCONFDIR - конфигурации (/etc)
#   CMAKE_INSTALL_SHAREDSTATEDIR - изменяемые данные (/var)
#   CMAKE_INSTALL_LOCALSTATEDIR - локальные данные (/var)
#   CMAKE_INSTALL_RUNSTATEDIR - runtime данные (/run)
#   CMAKE_INSTALL_LIBDIR - библиотеки (lib или lib64)
#   CMAKE_INSTALL_INCLUDEDIR - заголовки (include)
#   CMAKE_INSTALL_OLDINCLUDEDIR - старые заголовки (/usr/include)
#   CMAKE_INSTALL_DATAROOTDIR - архитектурно-независимые данные (share)
#   CMAKE_INSTALL_DATADIR - данные (share)
#   CMAKE_INSTALL_INFODIR - info документация (share/info)
#   CMAKE_INSTALL_LOCALEDIR - локализация (share/locale)
#   CMAKE_INSTALL_MANDIR - man страницы (share/man)
#   CMAKE_INSTALL_DOCDIR - документация (share/doc/PROJECT_NAME)
# ДЕТАЛИ: Учитывает CMAKE_INSTALL_PREFIX и платформенные особенности
# ВЕРСИЯ: Доступен с CMake 2.8.5
include(GNUInstallDirs)

# Установка библиотеки

# install() - правила установки файлов и таргетов.
# СИГНАТУРА для TARGETS: 
#   install(TARGETS targets... [EXPORT <export-name>]
#           [RUNTIME_DEPENDENCIES args...]
#           [<ARCHIVE|LIBRARY|RUNTIME|OBJECTS|FRAMEWORK|BUNDLE|
#             PRIVATE_HEADER|PUBLIC_HEADER|RESOURCE|FILE_SET <set>> 
#            [DESTINATION <dir>]
#            [PERMISSIONS permissions...]
#            [CONFIGURATIONS [Debug|Release|...]]
#            [COMPONENT <component>]
#            [NAMELINK_COMPONENT <component>]
#            [OPTIONAL] [EXCLUDE_FROM_ALL]
#            [NAMELINK_ONLY|NAMELINK_SKIP]
#           ] [...])
# ТИПЫ ФАЙЛОВ:
#   ARCHIVE - статические библиотеки и DLL import библиотеки (.a, .lib)
#   LIBRARY - shared библиотеки (.so, .dylib)
#   RUNTIME - исполняемые файлы и DLL (.exe, .dll)
#   OBJECTS - объектные файлы (.o)
#   FRAMEWORK - macOS frameworks
#   BUNDLE - macOS bundles
#   FILE_SET - наборы файлов (headers и т.д.)
# EXPORT - добавление в экспортируемый набор
# DESTINATION - путь установки (относительно CMAKE_INSTALL_PREFIX)
# COMPONENT - компонент для выборочной установки
# ВЕРСИЯ: Базовая с CMake 2.6, FILE_SET с 3.23
install(TARGETS ml_labs_core
    EXPORT ml_labs_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Экспорт CMake конфигурации для find_package(ml_labs)

# install(EXPORT) - установка файла с импортированными таргетами.
# СИГНАТУРА: install(EXPORT <export-name> DESTINATION <dir>
#                    [NAMESPACE <namespace>] [FILE <name>.cmake]
#                    [PERMISSIONS permissions...]
#                    [CONFIGURATIONS [Debug|Release|...]]
#                    [EXPORT_LINK_INTERFACE_LIBRARIES]
#                    [COMPONENT <component>]
#                    [EXCLUDE_FROM_ALL]
#                    [CXX_MODULES_DIRECTORY <dir>])
# ПАРАМЕТРЫ:
#   export-name - имя экспорта (из install(TARGETS ... EXPORT))
#   DESTINATION - директория для установки
#   NAMESPACE - префикс для импортированных таргетов
#   FILE - имя генерируемого файла
# ЭФФЕКТ: Создаёт файл с командами add_library(... IMPORTED)
# ВЕРСИЯ: Доступна с CMake 2.6
install(EXPORT ml_labs_targets
    FILE ml_labs-targets.cmake
    NAMESPACE ml_labs::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml_labs
)

# Генерация ml_labs-config.cmake

# include(CMakePackageConfigHelpers) - утилиты для создания пакетных конфигураций
# ПРЕДОСТАВЛЯЕТ:
#   configure_package_config_file() - генерация Config.cmake
#   write_basic_package_version_file() - генерация ConfigVersion.cmake
# ВЕРСИЯ: Доступен с CMake 2.8.8
include(CMakePackageConfigHelpers)

# configure_package_config_file() - генерация Config.cmake из шаблона.
# СИГНАТУРА: configure_package_config_file(<input> <output>
#                                           INSTALL_DESTINATION <path>
#                                           [PATH_VARS vars...]
#                                           [NO_SET_AND_CHECK_MACRO]
#                                           [NO_CHECK_REQUIRED_COMPONENTS_MACRO]
#                                           [INSTALL_PREFIX <path>])
# ПАРАМЕТРЫ:
#   input - путь к шаблону Config.cmake.in
#   output - путь к генерируемому Config.cmake
#   INSTALL_DESTINATION - путь установки конфигурации
#   PATH_VARS - переменные путей для подстановки
# ШАБЛОННЫЕ ПЕРЕМЕННЫЕ:
#   @PACKAGE_INIT@ - инициализация с правильными путями
#   @PACKAGE_VAR@ - подстановка переменных
# МАКРОСЫ В РЕЗУЛЬТАТЕ:
#   set_and_check() - установка с проверкой существования
#   check_required_components() - проверка REQUIRED компонентов
# ВЕРСИЯ: Доступна с CMake 2.8.8
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ml_labs-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ml_labs-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml_labs
)

# Версионирование

# write_basic_package_version_file() - создание ConfigVersion.cmake.
# СИГНАТУРА: write_basic_package_version_file(<filename>
#                                              [VERSION <major.minor.patch>]
#                                              [COMPATIBILITY <mode>]
#                                              [ARCH_INDEPENDENT])
# ПАРАМЕТРЫ:
#   filename - путь к генерируемому файлу
#   VERSION - версия пакета (по умолчанию PROJECT_VERSION)
#   COMPATIBILITY - режим совместимости версий
#   ARCH_INDEPENDENT - пакет не зависит от архитектуры
# РЕЖИМЫ СОВМЕСТИМОСТИ:
#   AnyNewerVersion - любая версия >= запрошенной
#   SameMajorVersion - та же мажорная версия, минорная >= запрошенной
#   SameMinorVersion - та же минорная версия, патч >= запрошенного
#   ExactVersion - точное совпадение версий
# ВЕРСИЯ: Доступна с CMake 2.8.8
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ml_labs-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# install(FILES) - установка обычных файлов.
# СИГНАТУРА: install(FILES files... 
#                    DESTINATION <dir>
#                    [PERMISSIONS permissions...]
#                    [CONFIGURATIONS [Debug|Release|...]]
#                    [COMPONENT <component>]
#                    [RENAME <name>] 
#                    [OPTIONAL] 
#                    [EXCLUDE_FROM_ALL])
# ПАРАМЕТРЫ:
#   FILES - список файлов для установки
#   DESTINATION - директория назначения
#   PERMISSIONS - права доступа (UNIX)
#   CONFIGURATIONS - конфигурации для установки
#   COMPONENT - компонент установки
#   RENAME - переименование при установке (только для одного файла)
#   OPTIONAL - не выдавать ошибку если файл не существует
# ВЕРСИЯ: Доступна с CMake 2.6
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ml_labs-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ml_labs-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ml_labs
)

# Вывод конфигурации для разработчика
# PROJECT_VERSION - автоматическая переменная из project(VERSION)
# CMAKE_CXX_COMPILER_ID - идентификатор компилятора (GNU, Clang, MSVC, etc.)
# CMAKE_CXX_COMPILER_VERSION - версия компилятора
# CMAKE_INSTALL_PREFIX - корневая директория установки
message(STATUS "")
message(STATUS "ML Labs Configuration:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build tests:      ${ML_LABS_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${ML_LABS_BUILD_BENCHMARKS}")
message(STATUS "  Build docs:       ${ML_LABS_BUILD_DOCS}")
message(STATUS "  Sanitizers:       ${ML_LABS_ENABLE_SANITIZERS}")
message(STATUS "  Coverage:         ${ML_LABS_ENABLE_COVERAGE}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
