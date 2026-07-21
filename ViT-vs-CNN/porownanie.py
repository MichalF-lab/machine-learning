"""
Lączy wyniki CIFAR-10 i Imagenette w jeden plik porownanie_wynikow.json.
Uruchom po Eksperymenty.py i Eksperymenty_ImageNet.py.
"""
import json


def load_results(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def experiment_key(r):
    return (r['model'], r['experiment'], r['param_value'])


def main():
    cifar = load_results('wyniki_eksperymentow.json')
    inet = load_results('wyniki_eksperymentow_imagenet.json')

    inet_map = {experiment_key(r): r for r in inet}

    print("=" * 90)
    print("ZESTAWIENIE: ViT vs CNN — CIFAR-10 vs Imagenette2-320 (resize 32x32)")
    print("=" * 90)

    # --- Tabela 1: Jakość klasyfikacji ---
    print("\n1. JAKOSC KLASYFIKACJI (Test Accuracy %)")
    headers = ['Model', 'Eksperyment', 'Epoki', 'CIFAR-10', 'Imagenette', 'Roznica']
    rows = []
    for c in cifar:
        key = experiment_key(c)
        i = inet_map.get(key, {})
        c_acc = c['test_acc']
        i_acc = i.get('test_acc', 0)
        diff = c_acc - i_acc
        rows.append([
            c['model'], c['experiment'], c['param_value'],
            f"{c_acc:.2f}%", f"{i_acc:.2f}%", f"{diff:+.2f}%"
        ])

    col_w = [max(len(str(r[j])) for r in rows + [headers]) for j in range(len(headers))]
    sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'
    hdr = '|' + '|'.join(f' {headers[j]:<{col_w[j]}} ' for j in range(len(headers))) + '|'
    print(sep); print(hdr); print(sep)
    for row in rows:
        print('|' + '|'.join(f' {str(row[j]):<{col_w[j]}} ' for j in range(len(row))) + '|')
    print(sep)

    # --- Tabela 2: Liczba parametrów ---
    print("\n2. LICZBA PARAMETROW")
    seen = set()
    rows2 = []
    for c in cifar:
        k = (c['model'], c['experiment'])
        if k not in seen:
            seen.add(k)
            rows2.append([c['model'], c['experiment'], f"{c['num_params']:,}", f"{c['num_params']/1e6:.2f}M"])

    h2 = ['Model', 'Konfiguracja', 'Parametry', 'Parametry (mln)']
    col_w2 = [max(len(str(r[j])) for r in rows2 + [h2]) for j in range(len(h2))]
    sep2 = '+' + '+'.join('-' * (w + 2) for w in col_w2) + '+'
    hdr2 = '|' + '|'.join(f' {h2[j]:<{col_w2[j]}} ' for j in range(len(h2))) + '|'
    print(sep2); print(hdr2); print(sep2)
    for row in rows2:
        print('|' + '|'.join(f' {str(row[j]):<{col_w2[j]}} ' for j in range(len(row))) + '|')
    print(sep2)

    # --- Tabela 3: Czas treningu ---
    print("\n3. KOSZT OBLICZENIOWY (sekundy)")
    h3 = ['Model', 'Eksperyment', 'Epoki', 'CIFAR-10 (s)', 'Imagenette (s)', 'C10/epoka', 'Inet/epoka']
    rows3 = []
    for c in cifar:
        key = experiment_key(c)
        i = inet_map.get(key, {})
        epochs = c['param_value']
        c_time = c['train_time_s']
        i_time = i.get('train_time_s', 0)
        rows3.append([
            c['model'], c['experiment'], epochs,
            f"{c_time:.1f}", f"{i_time:.1f}",
            f"{c_time/epochs:.1f}", f"{i_time/epochs:.1f}" if i_time > 0 else "N/A"
        ])

    col_w3 = [max(len(str(r[j])) for r in rows3 + [h3]) for j in range(len(h3))]
    sep3 = '+' + '+'.join('-' * (w + 2) for w in col_w3) + '+'
    hdr3 = '|' + '|'.join(f' {h3[j]:<{col_w3[j]}} ' for j in range(len(h3))) + '|'
    print(sep3); print(hdr3); print(sep3)
    for row in rows3:
        print('|' + '|'.join(f' {str(row[j]):<{col_w3[j]}} ' for j in range(len(row))) + '|')
    print(sep3)

    # --- Podsumowanie CNN 10ep vs Best ViT ---
    print("\n4. PODSUMOWANIE: CNN (10 epok) vs NAJLEPSZY ViT")
    cnn10_c = next(r for r in cifar if r['model'] == 'CNN' and r['param_value'] == 10)
    cnn10_i = next((r for r in inet if r['model'] == 'CNN' and r['param_value'] == 10), None)
    best_vit_c = max((r for r in cifar if r['model'] == 'ViT'), key=lambda x: x['test_acc'])
    best_vit_i = max((r for r in inet if r['model'] == 'ViT'), key=lambda x: x['test_acc']) if inet else None

    print(f"  CNN 10ep   | CIFAR-10: {cnn10_c['test_acc']:.2f}% ({cnn10_c['num_params']:,} params, {cnn10_c['train_time_s']:.0f}s)")
    if cnn10_i:
        print(f"  CNN 10ep   | Imagenette: {cnn10_i['test_acc']:.2f}% ({cnn10_i['num_params']:,} params, {cnn10_i['train_time_s']:.0f}s)")
    print(f"  Best ViT   | CIFAR-10: {best_vit_c['test_acc']:.2f}% [{best_vit_c['experiment']}] ({best_vit_c['num_params']:,} params, {best_vit_c['train_time_s']:.0f}s)")
    if best_vit_i:
        print(f"  Best ViT   | Imagenette: {best_vit_i['test_acc']:.2f}% [{best_vit_i['experiment']}] ({best_vit_i['num_params']:,} params, {best_vit_i['train_time_s']:.0f}s)")

    # Zapisz połączony plik
    comparison = {
        "cifar10": cifar,
        "imagenet": inet,
        "summary": {
            "cifar10": {
                "best_cnn": {"test_acc": cnn10_c['test_acc'], "params": cnn10_c['num_params'], "time_s": cnn10_c['train_time_s']},
                "best_vit": {"test_acc": best_vit_c['test_acc'], "params": best_vit_c['num_params'], "time_s": best_vit_c['train_time_s'], "config": best_vit_c['experiment']},
            },
            "imagenet": {
                "best_cnn": {"test_acc": cnn10_i['test_acc'], "params": cnn10_i['num_params'], "time_s": cnn10_i['train_time_s']} if cnn10_i else None,
                "best_vit": {"test_acc": best_vit_i['test_acc'], "params": best_vit_i['num_params'], "time_s": best_vit_i['train_time_s'], "config": best_vit_i['experiment']} if best_vit_i else None,
            }
        }
    }

    with open('porownanie_wynikow.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    print("\nPelne zestawienie zapisane do: porownanie_wynikow.json")


if __name__ == '__main__':
    main()
