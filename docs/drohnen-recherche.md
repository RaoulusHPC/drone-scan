# Drohnen-Recherche – drone-scan

> Stand: 20.11.2026 · Preise: Straßenpreise/UVP in Deutschland laut den Quellen unten, schwanken stark (Aktionen). Mit „≈“ markierte Werte sind eigene Schätzungen.
> Bezug: [Architektur](architektur.md), ADR-0004 (Drohnen-Agnostik), ADR-0005 (ArduPilot), Stories US-050, US-052 (Hardware).

## 1. Anforderungen an die Drohne

| Kriterium | Warum wichtig | Mindestanforderung |
|---|---|---|
| Videoqualität | SfM und Splat-Qualität hängen direkt an Schärfe, Auflösung und Stabilisierung | ≥ 2,7K, besser 4K, mechanischer Gimbal oder gute EIS, ≥ 30 fps |
| Indoor-Stabilität | Ohne GPS braucht man Sicht-/Flusssensoren nach unten, sonst driftet die Drohne | Vision Positioning / optischer Fluss |
| Kollisionsschutz | Enge Räume, Möbel | Propellerschutz Pflicht, Hinderniserkennung von Vorteil |
| Gewicht / Größe | Sicherheit, Türdurchgänge | < 250 g, Durchmesser < 25 cm |
| Flugzeit | Anzahl Teilflüge pro Wohnung | ≥ 10 min |
| Offenheit | Projektziel Open Source | **Minimal:** Video als Datei abrufbar. **Ideal:** offene Firmware (ArduPilot/PX4/Crazyflie) |
| Preis | Vorgabe | Entwicklung ≈ 100 €, Endnutzer ≤ 500 € |

**Wichtigste Erkenntnis:** Eine neue Drohne mit **guter Kamera und offener Firmware** gibt es unter 100 € derzeit nicht. Daraus folgt die Drei-Stufen-Strategie unten. Die Pipeline bleibt drohnen-agnostisch, sodass die Wahl der Drohne die Software nicht bindet.

## 2. Marktlage (Kurzfassung)

- **Programmierbare Billigdrohnen sind weggefallen:** Die Ryze/DJI Tello wurde 2018–2024 gebaut und ist eingestellt. Sie ist nur noch gebraucht bzw. als Restposten erhältlich.
- **Consumer-Klasse unter 250 g dominiert DJI.** Die Geräte sind gut für Innenräume, aber Firmware und SDK sind geschlossen. Für uns spielt das keine Rolle, solange die Videodatei verfügbar ist.
- **Offene Plattformen** sind entweder Forschungsplattformen (Bitcraze Crazyflie, teuer, schwache Kamera) oder Eigenbau-Cinewhoops mit ArduPilot plus Action-Cam.

## 3. Kandidaten im Vergleich

### 3.1 Consumer-Drohnen (geschlossen, „Videoquelle“)

| Modell | Preis (DE) | Gewicht | Video | Indoor / Sensorik | Flugzeit | Eignung für drone-scan |
|---|---|---|---|---|---|---|
| **DJI Neo 2** | UVP ab 239 €, Straße ≈ 209 € | 151–160 g (mit Propellerschutz) | 4K/60 (WLAN), 4K/100 (mit RC) | Rundum-Hinderniserkennung, Vollpropellerschutz, laut Herstellerangaben/Tests für Indoor-Flüge ausgelegt | bis 19 min | ⭐⭐⭐⭐ **Beste Entwicklungsdrohne im Budget-Rahmen**, typisches Endnutzergerät |
| DJI Neo (1. Gen.) | ≈ 189 € | ≈ 135 g | 4K/30 | Propellerschutz, keine Rundum-Sensorik | ≈ 18 min | ⭐⭐⭐ Günstiger, aber schwächer, kaum noch sinnvoll |
| DJI Mini 4K | UVP 299 €, Aktionen 209–235 € | < 249 g | 4K/30, 3-Achsen-Gimbal | **Keine Hinderniserkennung** (nur nach unten) | bis 31 min | ⭐⭐ Gute Kamera, indoor riskant (kein Propellerschutz ab Werk) |
| DJI Flip | ≈ 295 € (RC-N3-Variante, Aktion) | < 249 g | 4K | Integrierter Propellerschutz | – | ⭐⭐⭐ Indoor geeignet, etwas teurer |
| DJI Lito 1 / Lito X1 | 339 € / 419 € (X1 Aktion ≈ 409 €) | < 249 g | 4K/60, X1 mit 1/1,3″-Sensor | Rundum-Hinderniserkennung, X1 mit Front-LiDAR | bis 36 min | ⭐⭐⭐ Endnutzerklasse ≤ 500 €; offene Propeller → indoor nur mit Vorsicht |
| Ryze Tello (gebraucht) | ≈ 50–90 € (gebraucht, Schätzung) | ≈ 80 g | 720p, stark komprimiert | Vision Positioning nach unten | ≈ 13 min | ⭐⭐ Offene Python-Schnittstelle (djitellopy), aber schwache Bildqualität → nur für Adapter-Tests |

### 3.2 Offene Plattformen

| Plattform | Preis | Offenheit | Kamera | Eignung |
|---|---|---|---|---|
| **Eigenbau 3–3,5″-Cinewhoop mit ArduPilot** + Optical-Flow/ToF-Sensor + Action-Cam | ≈ 250–400 € (Schätzung inkl. Funk) | Firmware offen (ArduPilot, GPL), MAVLink, SITL | z. B. **RunCam Thumb 2**: 4K/60, 27 g, ≈ 100 USD, Gyro-Daten; leichter: Thumb Pro 4K V2 (16 g) | ⭐⭐⭐⭐ **Zielplattform für M5** (assistierter Flug) |
| Bitcraze Crazyflie 2.1 Brushless | ≈ 422 € netto (nur Drohne), AI-Bundle ≈ 780 USD | Hardware + Firmware offen | AI-Deck: 324×324 px | ⭐⭐ Hervorragend für Autonomie-Forschung, **ungeeignet für Kartierung** (Kamera), Nutzlast ≈ 40 g |

**Hinweise zur ArduPilot-Plattform**
- ArduPilot dokumentiert den **MatekSys 3901-L0X** (optischer Fluss plus Kurzstrecken-Lidar, MSP-Protokoll). Im Forum gibt es allerdings Berichte über schlechte Flussqualität indoor. Ein Nutzer berichtet mit dem **MicoAir MTF-02** von besseren Ergebnissen. Empfehlung: MTF-01/02 bevorzugen und beide in US-052 testen.
- Für Navigation ohne GPS braucht der Flight Controller nach Forenhinweisen **mehr als 1 MB Flash**. Also F405/H743-Klasse wählen, keine F411-Boards.
- Bodenstation auf Android: **QGroundControl** (Open Source).

## 4. Empfehlung je Projektphase

| Phase | Empfehlung | Kosten | Begründung |
|---|---|---|---|
| **M0–M2** (Pipeline, Suche) | **Smartphone** als Videoquelle | 0 € | Beste Bildqualität bei null Kosten. Trennt Software-Risiken von Flug-Risiken |
| **M3** (Drohnen-Erfassung) | **DJI Neo 2** (Basispaket ohne RC) | ≈ 209–239 € | Indoor-tauglich (Propellerschutz, Rundum-Sensorik), 4K/60, repräsentativ für Endnutzergeräte. Liegt über dem 100-€-Ziel, aber **unter 100 € gibt es keine neue Drohne mit brauchbarer Kamera** |
| M3, Sparvariante | Gebrauchte **Tello** | ≈ 50–90 € | Nur für Adapter-Entwicklung (US-017). Für Qualitätsaussagen ungeeignet |
| **M5** (assistierter Flug) | **3–3,5″-Cinewhoop mit ArduPilot**, F405/H743-FC, MicoAir MTF-01/02, RunCam Thumb 2 | ≈ 250–400 € | Einzige Option mit offener Firmware **und** guter Kamera im Budget |
| Endnutzer-Referenzliste | Neo 2, Flip, Lito 1 / X1 (mit Vorsicht indoor), Mini 4K (nur mit Propellerschutz/Erfahrung) | ≤ 500 € | Kameraprofile in US-014 / US-050 anlegen |

**Kaufzeitpunkt:** DJI Neo 2 in Sprint 5 (Mitte Februar 2027), damit sie zu Beginn von M3 bereitsteht. ArduPilot-Komponenten in Sprint 9.

## 5. Rechtliches & Sicherheit (Kurz)

- **Indoor:** Für Drohnen, die ausschließlich in Innenräumen fliegen, ist laut gängiger Darstellung kein Drohnenführerschein nötig. Vor gewerblicher Nutzung solltest du das juristisch bestätigen lassen.
- **Haftpflicht:** Für Drohnen allgemein vorgeschrieben (§ 43 Abs. 2 LuftVG). Prüfen, ob die Police auch Indoor- und gewerbliche Nutzung abdeckt.
- **Neo 2 / Mini 4K / Lito** liegen unter 250 g (EU-Klasse C0) und sind damit auch im Freien einfach zu betreiben (Registrierung als Betreiber bei Kamera nötig).
- **Sicherheit:** Propellerschutz, langsames Fliegen, keine Personen/Tiere (Projektannahme), Not-Aus beim Eigenbau.

## 6. Offene Punkte (in US-050 zu klären)

1. Liefert die Neo 2 Videos mit ausreichend wenig Bewegungsunschärfe bei Wohnungslicht? Verhält sich ihr Auto-Weißabgleich stabil? (Tests nennen Schwächen beim Weißabgleich.)
2. Wie groß ist der Unterschied zwischen 4K/60 über WLAN und 4K/100 mit Fernsteuerung für die Rekonstruktion?
3. Wie zuverlässig liefert der optische Fluss (MTF-01/02 vs. 3901-L0X) die Positionshaltung über typischen Böden (Parkett, Fliesen, Teppich)?

## Quellen

- Tello eingestellt (2018–2024): https://en.wikipedia.org/wiki/Ryze_Tello
- DJI Neo 2, Preis ab 239 €: https://www.notebookcheck.com/DJI-Neo-2-Kompakte-Drohne-hebt-mit-4K-Zeitlupenaufnahmen-und-Gesten-und-Sprachsteuerung-ab.1161237.0.html
- DJI Neo 2, Specs/Indoor: https://www.drohnen.de/70098/dji-neo-2-test/ · https://www.calumet.de/magazin/dji-neo-2-dein-fliegender-kameramann · Preisvergleich: https://www.testberichte.de/p/dji-tests/neo-2-testbericht.html
- DJI Neo 2 Test (Weißabgleich): https://www.notebookcheck.com/Kamera-Drohne-DJI-Neo-2-mit-Mini-Display-im-Test-Was-die-Neo-1-haette-sein-sollen.1155772.0.html
- DJI Neo / Neo 2 Preise Amazon: https://www.pcgameshardware.de/Drohne-Hardware-279812/News/DJI-Neo-2-4K-Foto-Video-Drohne-Sprachsteuerung-Smartwatch-Tiefstpreis-Angebot-guenstig-1535372/
- DJI Mini 4K Preise: https://www.drohnen.de/50714/dji-mini-4k-preise/ · https://www.macwelt.de/article/3030536/dji-mini-bestseller-drohne-mit-4k-uhd-kamera-und-gimbal-bestpreis-amzon.html
- DJI Mini 4K vs. Flip (Okt. 2026): https://www.drohnen.de/?p=77264
- DJI Lito 1 / X1: https://www.notebookcheck.com/DJI-Lito-1-und-Lito-X1-heben-als-einsteigerfreundliche-Drohnen-mit-4K-Video-ab.1278923.0.html · https://www.drohnen.de/83025/dji-lito-test/ · https://www.pcwelt.de/article/3185566/dji-lito-x1-drohne-tiefstpreis-amazon-deal.html
- Crazyflie 2.1 Brushless: https://exp-tech.de/en/products/crazyflie-2-1-brushless · https://store.bitcraze.io/collections/bundles-crazyflie-2-1-brushless/products/the-ai-bundle-crazyflie-2-1-brushless · https://www.unmannedtechshop.co.uk/products/crazyflie-2-1-brushless-hacker-drone · AI-Deck-Auflösung: https://www.bitcraze.io/tag/ai-deck/
- ArduPilot Optical Flow: https://ardupilot.org/copter/docs/common-mateksys-optflow-3901L0X.html · https://ardupilot.org/rover/docs/common-mtf-01.html · Forum: https://discuss.ardupilot.org/t/optical-flow-and-rangefinder-need-position-estimate/142210 · Flash-Hinweis: https://discuss.ardupilot.org/t/config-error-baro-goku-gn745-aio/98264?page=3
- RunCam Thumb 2: https://shop.runcam.com/runcam-thumb-2 · https://www.getfpv.com/action-cameras/runcam-thumb-2-4k-hd-action-camera.html · Thumb Pro 4K V2: https://www.racedayquads.com/products/runcam-thumb-pro-4k-v2-hd-action-camera-gyroflow-compatible
- Indoor-Flug / Haftpflicht: https://www.pcwelt.de/?p=1151724 · https://www.test.de/drohnen-recht
