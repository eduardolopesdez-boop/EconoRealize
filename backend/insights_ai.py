def gerar_insight(modelo, variavel="selic_mensal"):
    coef = modelo.params.get(variavel, 0)
    r2 = modelo.rsquared * 100
    if coef > 0:
        rel = "aumenta"
    else:
        rel = "reduz"
    return (
        f"A cada +1 p.p. na {variavel.replace('_', ' ')}, "
        f"a inadimplência tende a {rel} em {abs(coef):,.0f} milhões. "
        f"O modelo explica {r2:.1f}% da variação observada."
    )
