"""
Test sprawdzający czy additional_hf_datasets zwraca próbki z kluczem 'content'
"""

from src.scriptguard.data_sources.additional_hf_datasets import AdditionalHFDatasets

def test_fallback_samples():
    """Test czy fallback samples mają klucz 'content'"""
    hf = AdditionalHFDatasets()

    # Test fallback malware samples
    fallback_samples = hf._generate_fallback_malware_samples(count=3, source="test")

    print("✓ Testowanie fallback malware samples...")
    for i, sample in enumerate(fallback_samples):
        assert "content" in sample, f"Sample {i} nie ma klucza 'content'!"
        assert "code" not in sample, f"Sample {i} ma stary klucz 'code'!"
        assert "label" in sample, f"Sample {i} nie ma klucza 'label'!"
        assert "source" in sample, f"Sample {i} nie ma klucza 'source'!"
        assert len(sample["content"]) > 50, f"Sample {i} ma za krótki content!"

    print(f"  ✅ {len(fallback_samples)} fallback samples - OK")

    # Test fallback C2 samples
    c2_samples = hf._generate_fallback_c2_samples(count=2)

    print("✓ Testowanie fallback C2 samples...")
    for i, sample in enumerate(c2_samples):
        assert "content" in sample, f"C2 Sample {i} nie ma klucza 'content'!"
        assert "code" not in sample, f"C2 Sample {i} ma stary klucz 'code'!"
        assert "label" in sample, f"C2 Sample {i} nie ma klucza 'label'!"
        assert "source" in sample, f"C2 Sample {i} nie ma klucza 'source'!"
        assert len(sample["content"]) > 50, f"C2 Sample {i} ma za krótki content!"

    print(f"  ✅ {len(c2_samples)} C2 samples - OK")

    # Sprawdź strukturę
    example = fallback_samples[0]
    print("\n✓ Przykładowa struktura sample:")
    print(f"  Keys: {list(example.keys())}")
    print(f"  Label: {example['label']}")
    print(f"  Source: {example['source']}")
    print(f"  Content length: {len(example['content'])} chars")
    print(f"  Content preview: {example['content'][:100]}...")

    print("\n" + "="*60)
    print("✅ WSZYSTKIE TESTY PRZESZŁY!")
    print("="*60)
    print("\n📊 Podsumowanie:")
    print(f"  - Wszystkie samples mają klucz 'content' ✓")
    print(f"  - Żaden sample nie ma starego klucza 'code' ✓")
    print(f"  - Struktura jest zgodna z pipeline ✓")
    print("\n🎉 Poprawka działa! Teraz statystyki będą pokazywać wszystkie źródła.")

if __name__ == "__main__":
    test_fallback_samples()
