#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para gerar o projeto completo
Chama generate1.py, generate2.py e generate3.py
"""

from generate1 import run_generate1
from generate2 import run_generate2
from generate3 import run_generate3


def main():
    print("=" * 70)
    print("GERADOR DE PROJETO - SISTEMA DE ANALISE ECONOMETRICA")
    print("=" * 70)
    print()
    
    # Parte 1: Cria pasta e app.py
    run_generate1()
    print()
    
    # Parte 2: Cria auth.py e models.py
    run_generate2()
    print()
    
    # Parte 3: Cria visualizations.py, export.py, requirements.txt e README.md
    run_generate3()
    print()
    
    print("=" * 70)
    print("PROJETO CRIADO COM SUCESSO!")
    print("=" * 70)
    print()
    print("Arquivos criados em: econometric_system/")
    print("  - app.py")
    print("  - auth.py")
    print("  - models.py")
    print("  - visualizations.py")
    print("  - export.py")
    print("  - requirements.txt")
    print("  - README.md")
    print()
    print("Para executar:")
    print("  1. cd econometric_system")
    print("  2. pip install -r requirements.txt")
    print("  3. streamlit run app.py")
    print()
    print("Login: admin / admin123")
    print()


if __name__ == "__main__":
    main()
