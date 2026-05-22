# Streamlit Anomaly Detection Project
### Abdullah Yağız ÖZBEK

## Genel Bakış
1. TUSAŞ tarafından verilen case study anomaly detection projesi.
2. Proje StreamLit üzerinden deploy olduğu için kullanıcıya erişim https://casestudy-anomaly-i48z01ds9j.streamlit.app/ adresinden kolayca erişebilir.
3. Streamlit, github üzerinden bağlı olduğu repository i deploy eder ve web sitesinden erişilebilir hale gelir.
4. URL uzun süre kullanılmaz ise cloud sistemi uyku moduna girebilir, bunun için uyandırmanız lazım. 2-3 dk sürebiliyor projenin açılması.

## Kullanım Talimatları
1. Resimler yüklenmeden önce altta belirtilen işlemler öncesinde yapılmalıdır.
    a. "Select product category" drop box içerisinde yüklenecek ürün türü seçilmelidir. Bunlar Hazelnut ve Wood. Seçilen ürüne göre train edilen model seçilmektedir.
    b. Threshold slider kullanılarak değeri değiştirilebilir. Default olarak optimal değerler Hazelnut ve Wood için belirlenmiştir.
    c. "Default Ground Truth Label" drop box ile kullanıcıya rahatlık sağlaması için batch olarak yüklenen resimlere ground-truth değeri set edilir. Böylece True-positive
        True-negative gibi metrikler hesaplanabilir.
    d. Sonuçlar oluşturulduktan sonra, her resim için ayrı ayrı ground-truth değeri tekrar set edilebilir.
2. Resimler yüklendikten sonra program:
    a. Yüklenen resimleri işler.
    b. PatchCore anomali tahminlerini yapar.
    c. Anomaly heatmap üretilir.
    d. sonuçlar ekranda gösterilir.
    e. Sayfanın en altına metrik sonuçlarını basar.