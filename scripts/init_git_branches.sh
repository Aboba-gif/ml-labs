#!/bin/bash
# scripts/init_git_branches.sh

echo "Setting up Git Flow branch structure..."

# Основная ветка - main
git branch -M main

# Ветка разработки
git checkout -b develop

# Feature branches для компонентов
git checkout -b feature/tensor-simd develop
git checkout -b feature/autodiff-tape develop
git checkout -b feature/optimizer-adam develop

# Ветки для лабораторных
git checkout -b lab/01-linear-regression develop
git checkout -b lab/02-logistic-regression develop
git checkout -b lab/03-neural-network develop

# Experimental ветки для исследований
git checkout -b experimental/cuda-backend develop
git checkout -b experimental/opencl-backend develop

# Возвращаемся в develop
git checkout develop

echo "Branch structure created!"
echo ""
echo "Current branches:"
git branch -a
