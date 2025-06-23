import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


disease_info = {
    "Common rust": {
        "symptoms": "Minute flecks appear on both sides of the leaves and slowly develop into small, tan, slightly raised spots. These mainly elongated spots later turn into powdery, golden-brown pustules loosely scattered in patches on upper and lower sides. The color can change to black as the plant matures. Contrarily to other rust diseases, symptoms are not usually present on other plant parts, such as stalks, sheath leaves or husks. However, stalks tend to grow weak and soft and are prone to lodging.",
        "chemical_control": "Apply a foliar fungicide early in the season if rust is bound to spread rapidly. Products containing mancozeb, pyraclostrobin, pyraclostrobin + metconazole, pyraclostrobin + fluxapyroxad, azoxystrobin + propiconazole, trifloxystrobin + prothioconazole can be used. Recommendation: Spray mancozeb @ 2.5 g/l as soon as pustules appear and repeat at 10-day intervals until flowering.",
        "organic_control": "",
        "preventive_measures": [
            "Plant resistant varieties available locally",
            "Plant early to avoid optimal conditions for infection",
            "Use shorter season varieties that mature earlier",
            "Monitor your crop regularly, especially during overcast weather",
            "Ensure balanced fertilization with split applications of nitrogen"
        ],
        "cause": "The disease is caused by a fungus that produces spores that are distributed by wind or rain."
    },
    "Cercospora leaf spot Gray leaf spot": {
        "symptoms": "Small necrotic (brown or tan) spots that may have a yellow chlorotic halo appear on lower leaves usually before flowering. Gradually these lesions will turn grayish and appear on younger leaves too. As the disease progresses, they enlarge into elongated, rectangular lesions that run parallel to the leaf veins. In optimal conditions, they can coalesce and engulf the whole leaf.",
        "chemical_control": "Foliar fungicide treatment is effective if applied at early stages. Fungicides containing pyraclostrobin and strobilurin, or combinations of azoxystrobin and propiconazole, prothioconazole and trifloxystrobin work well to control the fungus.",
        "organic_control": "No biological control is available to control this disease.",
        "preventive_measures": [
            "Plant resistant varieties if available in your area",
            "Plant late to avoid adverse conditions for plants",
            "Keep up good ventilation by widening the space between plants",
            "Plow deep and bury all plant residues after harvest",
            "Plan long-term crop rotations with non-host plants"
        ],
        "cause": "The gray leaf spot disease is caused by the fungus Cercospora zeae-maydis. It survives in plant residues in the soil for long periods of time. Its lifecycle is favored by elevated temperatures (25 to 30°C), high humidity and leaf wetness for prolonged periods of time."
    },
    "Northern Leaf Blight": {
        "symptoms": "Tan, diamond-shaped to elongated lesions with brownish margins appear first on lower leaves and then slowly move up to younger foliage. Lesions are of different sizes and they extend beyond the leaf veins in susceptible plants. Lesions may coalesce which results in complete blight of large parts of the leaves.",
        "chemical_control": "Consider application after weighing disease development against potential yield loss and weather forecast. Fast-acting, broad spectrum products recommended such as mancozeb (2.5 g/l) at 8-10 days interval.",
        "organic_control": "Biocontrol with the competitive fungus Trichoderma atroviride has shown promising results in controlled environments, though field trials are still needed.",
        "preventive_measures": [
            "Plant resistant varieties if available in your area",
            "Plant different varieties of maize to avoid monocultures",
            "Keep field clean",
            "Rotate with non-host crops",
            "Plow deep to bury crop residues in the soil",
            "Plan a fallow after the harvest"
        ],
        "cause": "The disease is caused by the fungus Cochliobolus heterostrophus (also known as Bipolaris maydis). The fungus survives in plant residues in the soil and produces spores that are distributed by wind and rain splashes. The development is favored by moist weather, leaf wetness and temperatures ranging from 22 to 30°C."
    },
    "Healthy": {
        "symptoms": "No symptoms of disease. The plant appears normal with healthy green leaves.",
        "chemical_control": "No chemical control needed for healthy plants.",
        "organic_control": "",
        "preventive_measures": [
            "Continue regular monitoring for early signs of disease",
            "Maintain proper irrigation - avoid overwatering",
            "Ensure adequate nutrients through balanced fertilization",
            "Practice crop rotation to maintain soil health",
            "Remove any weeds that might compete for resources"
        ],
        "cause": "Your crop is healthy!"
    }
}

def get_model(path):
    model = load_model(path, compile=False)
    return model

def img_predict(path, crop):
    data = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(data).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255
    #model = get_model(r'C:\Users\rohit\OneDrive\Desktop\AgriGo-main\Farmer Assistant\models\DL_models\corn_model.h5')
    model = get_model("models/DL_models/corn_model.h5")
    if len(crop_diseases_classes[crop]) > 2:
        predicted = np.argmax(model.predict(data)[0])
    else:
        p = model.predict(data)[0]
        predicted = int(np.round(p)[0])
    return predicted


# Then modify your get_diseases_classes function to also return the disease info
def get_diseases_classes(crop, prediction):
    crop_classes = crop_diseases_classes[crop]
    disease_name = crop_classes[prediction][1].replace("_", " ")
    # Find the matching disease key in the disease_info dictionary
    for key in disease_info:
        if key in disease_name:
            disease_data = disease_info[key]
            return disease_name, disease_data
    return disease_name, {}  # Return empty dict if no matching disease info found

crop_details = {
    "apple": {
        "name": "Apple",
        "devnagri_name": "सफरचंद",
        "image_url": "https://example.com/apple.jpg",
        "description": "Apple is a deciduous fruit tree belonging to the Rosaceae family. It is cultivated worldwide for its sweet, crisp fruits. Apples are rich in dietary fiber, vitamin C, and various antioxidants. The tree typically grows 3-12 meters tall with spreading branches. Fruits vary in color from red, green to yellow, depending on the variety. Apples are consumed fresh, used in cooking, baking, and processed into juice, cider, and vinegar. They are economically important in temperate regions like Himachal Pradesh and Jammu and Kashmir in India.",
        "growing_season": "Spring to Fall (September-November in India)",
        "water_requirements": 800,
        "soil_requirements": "Well-drained, slightly acidic soil (pH 6.0-7.0)",
        "common_varieties": ["Red Delicious", "Gala", "Fuji", "Golden Delicious"],
        "major_growing_regions": ["Himachal Pradesh", "Jammu and Kashmir", "Uttarakhand"],
        "cultivation_tips": "Requires cold climate, annual pruning, pest management, and controlled irrigation"
    },
    "banana": {
        "name": "Banana",
        "devnagri_name": "केळ",
        "image_url": "https://example.com/banana.jpg",
        "description": "Banana is a large herbaceous flowering plant in the Musaceae family, native to tropical regions of Southeast Asia. It is one of the world's most important food crops, providing nutrition and economic sustenance. The plant produces large hanging clusters of edible fruits. Bananas are rich in potassium, vitamin B6, vitamin C, and dietary fiber. They are consumed fresh, used in cooking, baking, and processed into various products like chips and flour. India is the world's largest banana producer, with significant cultivation in Maharashtra, Tamil Nadu, and Karnataka.",
        "growing_season": "Year-round in tropical regions",
        "water_requirements": 1800,
        "soil_requirements": "Well-drained, fertile soil (pH 5.5-6.5)",
        "common_varieties": ["Cavendish", "Robusta", "Nendran"],
        "major_growing_regions": ["Maharashtra", "Tamil Nadu", "Karnataka"],
        "cultivation_tips": "Requires warm temperatures, high humidity, consistent moisture, and protection from strong winds"
    },
    "blackgram": {
        "name": "Black Gram",
        "devnagri_name": "उडीद",
        "image_url": "https://example.com/blackgram.jpg",
        "description": "Black Gram, also known as Urad dal, is a leguminous crop belonging to the Fabaceae family. It is an important pulse crop in India, primarily cultivated for its protein-rich seeds. The plant is an annual, erect herb growing 30-50 cm tall with compound leaves and white or pale purple flowers. Black gram is a crucial ingredient in Indian cuisine, used to make dal, vada, and other traditional dishes. It is nitrogen-fixing, improving soil fertility. The crop is drought-tolerant and grows well in various soil types.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 600,
        "soil_requirements": "Well-drained, sandy loam soil (pH 6.0-7.5)",
        "common_varieties": ["T-9", "PU-19", "LBG-752"],
        "major_growing_regions": ["Madhya Pradesh", "Maharashtra", "Andhra Pradesh"],
        "cultivation_tips": "Requires minimal irrigation, tolerant to drought, needs proper seed treatment and inoculation"
    },
    "chickpea": {
        "name": "Chickpea",
        "devnagri_name": "हरभरा",
        "image_url": "https://example.com/chickpea.jpg",
        "description": "Chickpea, also known as Gram or Chana, is a legume crop of significant economic and nutritional importance. It is an annual plant with small, feathery leaves and white or pinkish flowers. Chickpeas are rich in protein, fiber, vitamins, and minerals. They are used extensively in global cuisines, particularly in Indian, Middle Eastern, and Mediterranean cooking. The crop is drought-resistant and can grow in various soil types. India is the world's largest producer of chickpeas, with major cultivation in states like Madhya Pradesh and Maharashtra.",
        "growing_season": "Rabi (October-March)",
        "water_requirements": 500,
        "soil_requirements": "Well-drained, clay loam soil (pH 6.0-7.5)",
        "common_varieties": ["Kabuli", "Desi", "ICCC-37"],
        "major_growing_regions": ["Madhya Pradesh", "Maharashtra", "Rajasthan"],
        "cultivation_tips": "Requires moderate irrigation, needs protection from frost, benefits from crop rotation"
    },
    "coconut": {
        "name": "Coconut",
        "devnagri_name": "नारळ",
        "image_url": "https://example.com/coconut.jpg",
        "description": "Coconut is a versatile palm tree belonging to the Arecaceae family, grown in tropical and subtropical regions. It is known for its multiple uses in food, medicine, and industry. The tree produces large, green drupes that mature into brown, fibrous nuts containing nutritious white flesh and water. Coconuts are rich in medium-chain triglycerides, minerals, and electrolytes. Every part of the coconut is utilized - water, meat, oil, shell, and fiber. Kerala and Tamil Nadu are major coconut-producing states in India.",
        "growing_season": "Perennial (continuous production)",
        "water_requirements": 2000,
        "soil_requirements": "Sandy loam, well-drained coastal soils (pH 5.5-7.0)",
        "common_varieties": ["Tall", "Dwarf", "Hybrid"],
        "major_growing_regions": ["Kerala", "Tamil Nadu", "Karnataka"],
        "cultivation_tips": "Requires high humidity, consistent moisture, protection from strong winds, and regular fertilization"
    },
    "coffee": {
        "name": "Coffee",
        "devnagri_name": "कॉफी",
        "image_url": "https://example.com/coffee.jpg",
        "description": "Coffee is a tropical evergreen shrub or tree cultivated for its seeds, which are processed to produce the popular beverage. There are two primary species: Arabica and Robusta. The plant produces white, fragrant flowers and red or purple fruits containing seeds (coffee beans). Coffee is grown in regions with specific climatic conditions, typically at higher elevations. Karnataka, Kerala, and Tamil Nadu are major coffee-producing states in India. The crop is economically significant and supports millions of farmers worldwide.",
        "growing_season": "Perennial (continuous production)",
        "water_requirements": 1500,
        "soil_requirements": "Well-drained, slightly acidic soil (pH 6.0-6.5)",
        "common_varieties": ["Arabica", "Robusta", "Liberica"],
        "major_growing_regions": ["Karnataka", "Kerala", "Tamil Nadu"],
        "cultivation_tips": "Requires shade, consistent moisture, protection from direct sunlight, and careful pruning"
    },
    "cotton": {
        "name": "Cotton",
        "devnagri_name": "कापूस",
        "image_url": "https://example.com/cotton.jpg",
        "description": "Cotton is a soft, fluffy staple fiber that grows in a boll around the seeds of cotton plants. It is a crucial cash crop belonging to the Malvaceae family. The plant is an annual shrub with large leaves and beautiful flowers that develop into cotton bolls. Cotton is primarily cultivated for its fiber used in textile manufacturing, but seeds are also used for oil production. India is one of the world's largest cotton producers, with significant cultivation in Maharashtra, Gujarat, and Telangana. The crop plays a vital role in the agricultural economy and textile industry.",
        "growing_season": "Kharif (June-October)",
        "water_requirements": 900,
        "soil_requirements": "Well-drained, black or alluvial soil (pH 6.0-7.5)",
        "common_varieties": ["Bt Cotton", "Hybrid Cotton", "Desi Cotton"],
        "major_growing_regions": ["Maharashtra", "Gujarat", "Telangana"],
        "cultivation_tips": "Requires warm climate, moderate irrigation, pest management, and careful fertilization"
    },
    "grapes": {
        "name": "Grapes",
        "devnagri_name": "द्राक्ष",
        "image_url": "https://example.com/grapes.jpg",
        "description": "Grapes are berry fruits of the woody vine plants in the Vitaceae family. They are cultivated worldwide for fresh consumption, wine production, and dried fruit (raisins). The plant produces clusters of small, round fruits in various colors including green, red, and purple. Grapes are rich in antioxidants, vitamins, and minerals. Maharashtra's Nashik region is known as the 'Grape Capital of India', with significant cultivation and wine production. Grapes are used in multiple culinary applications, from fresh eating to wine and juice production.",
        "growing_season": "Rabi (October-March)",
        "water_requirements": 750,
        "soil_requirements": "Well-drained, sandy loam soil (pH 6.0-6.5)",
        "common_varieties": ["Thompson Seedless", "Flame Seedless", "Sharad Seedless"],
        "major_growing_regions": ["Maharashtra", "Karnataka", "Andhra Pradesh"],
        "cultivation_tips": "Requires trellising, pruning, controlled irrigation, and protection from extreme temperatures"
    },
    "jute": {
        "name": "Jute",
        "devnagri_name": "जुट",
        "image_url": "https://example.com/jute.jpg",
        "description": "Jute is a long, soft, shiny vegetable fiber that can be spun into coarse, strong threads. It is primarily used to make burlap, hessian, or gunny sacks. The plant is an annual crop belonging to the Corchorus genus, grown in tropical and subtropical regions. West Bengal is the largest jute-producing state in India. The crop is environmentally friendly, biodegradable, and has multiple industrial applications including packaging, textiles, and agricultural uses. Jute plays a significant role in the rural economy of eastern India.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 1000,
        "soil_requirements": "Fertile, clay loam soil with good water retention (pH 6.0-7.0)",
        "common_varieties": ["JRO-524", "JRO-7835", "Swarnamukhi"],
        "major_growing_regions": ["West Bengal", "Bihar", "Assam"],
        "cultivation_tips": "Requires high humidity, consistent moisture, and careful harvesting techniques"
    },
    "kidneybeans": {
        "name": "Kidney Beans",
        "devnagri_name": "राजमा",
        "image_url": "https://example.com/kidneybeans.jpg",
        "description": "Kidney beans are a variety of the common bean, named for their distinctive kidney-like shape. They are an important legume crop rich in protein, fiber, vitamins, and minerals. The plant is an annual herb with compound leaves and white or pink flowers. Kidney beans are a staple in many cuisines, particularly in North Indian cooking. They are nitrogen-fixing crops that improve soil fertility. The beans are used in various dishes, from curries to salads, and are valued for their nutritional benefits.",
        "growing_season": "Rabi (October-March)",
        "water_requirements": 600,
        "soil_requirements": "Well-drained, fertile soil (pH 6.0-7.0)",
        "common_varieties": ["Pusa Ratna", "Arka Sonali", "ICPL-87"],
        "major_growing_regions": ["Jammu and Kashmir", "Himachal Pradesh", "Uttarakhand"],
        "cultivation_tips": "Requires moderate irrigation, protection from frost, and crop rotation"
    },
    "lentil": {
        "name": "Lentil",
        "devnagri_name": "मसूर",
        "image_url": "https://example.com/lentil.jpg",
        "description": "Lentils are edible legumes known for their lens-shaped seeds. They are an important protein source in vegetarian diets, rich in nutrients and easy to digest. The plant is an annual bush with small flowers producing pods containing the lentil seeds. India is the world's largest producer and consumer of lentils. They are used in various culinary preparations like dal, soups, and stews. Lentils are drought-resistant and can grow in diverse soil conditions, making them an important crop for food security.",
        "growing_season": "Rabi (October-March)",
        "water_requirements": 450,
        "soil_requirements": "Well-drained, clay loam soil (pH 6.0-7.5)",
        "common_varieties": ["Pusa Badam", "Pusa-4", "L-4076"],
        "major_growing_regions": ["Madhya Pradesh", "Maharashtra", "Uttar Pradesh"],
        "cultivation_tips": "Requires minimal irrigation, benefits from crop rotation, needs careful seed treatment"
    },
    "maize": {
        "name": "Maize",
        "devnagri_name": "मका",
        "image_url": "https://example.com/maize.jpg",
        "description": "Maize, also known as corn, is a cereal grain first domesticated in Mexico. It is a versatile crop used for human consumption, animal feed, and industrial products. The plant produces large ears containing kernels on a central cob. Maize is rich in carbohydrates, vitamins, and minerals. India is among the top maize-producing countries, with significant cultivation in Karnataka, Madhya Pradesh, and Maharashtra. The crop has multiple uses in food, feed, and industrial applications like biofuel production.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 800,
        "soil_requirements": "Well-drained, fertile soil (pH 6.0-7.0)",
        "common_varieties": ["Hybrid Maize", "Deccan-103", "HQPM-1"],
        "major_growing_regions": ["Karnataka", "Madhya Pradesh", "Maharashtra"],
        "cultivation_tips": "Requires consistent moisture, proper spacing, and balanced fertilization"
    },
    "mango": {
        "name": "Mango",
        "devnagri_name": "आंबा",
        "image_url": "https://example.com/mango.jpg",
        "description": "Mango is a tropical fruit tree belonging to the Anacardiaceae family, known as the 'King of Fruits'. It produces large, sweet fruits with a distinctive flavor and aroma. The tree is evergreen, growing 30-100 feet tall with dense foliage. India is the world's largest mango producer, with significant cultivation in Maharashtra, Uttar Pradesh, and Andhra Pradesh. Mangoes are consumed fresh, processed into juices, pickles, and other products. They are rich in vitamins A and C, and have cultural significance in Indian cuisine and traditions.",
        "growing_season": "Summer (February-June)",
        "water_requirements": 1200,
        "soil_requirements": "Well-drained, deep soil (pH 5.5-7.5)",
        "common_varieties": ["Alphonso", "Dasheri", "Langra"],
        "major_growing_regions": ["Maharashtra", "Uttar Pradesh", "Andhra Pradesh"],
        "cultivation_tips": "Requires pruning, pest management, and careful irrigation"
    },
    "mothbeans": {
        "name": "Moth Beans",
        "devnagri_name": "मठ",
        "image_url": "https://example.com/mothbeans.jpg",
        "description": "Moth beans, also known as Matki, are a drought-resistant legume crop native to India. The plant is an annual herb with small, compound leaves and produces small, brown seeds. Moth beans are rich in protein, minerals, and dietary fiber. They are primarily cultivated in dry and semi-arid regions of Maharashtra, Rajasthan, and Gujarat. The beans are used in various culinary preparations, sprouts, and as animal feed. They are known for their ability to improve soil fertility through nitrogen fixation.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 400,
        "soil_requirements": "Sandy or light soil (pH 6.0-7.5)",
        "common_varieties": ["BINA-9", "BINA-6", "IGMV-99-1"],
        "major_growing_regions": ["Maharashtra", "Rajasthan", "Gujarat"],
        "cultivation_tips": "Extremely drought-tolerant, requires minimal irrigation, benefits from intercropping"
    },
    "mungbean": {
        "name": "Mung Bean",
        "devnagri_name": "मुग",
        "image_url": "https://example.com/mungbean.jpg",
        "description": "Mung bean is a small, green legume widely cultivated across Asia, particularly in India. It is a fast-growing annual plant known for its high protein content and easy digestibility. The beans are used in various culinary preparations, sprouts, and as a nutritious food source. Mung beans are drought-resistant, nitrogen-fixing, and play a crucial role in crop rotation. They are rich in essential nutrients like protein, fiber, vitamins, and minerals. The crop is economically important in agricultural systems, particularly in rain-fed and marginal farming regions.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 500,
        "soil_requirements": "Well-drained, sandy loam soil (pH 6.0-7.5)",
        "common_varieties": ["SML-668", "TARM-1", "DBMR-1"],
        "major_growing_regions": ["Maharashtra", "Madhya Pradesh", "Andhra Pradesh"],
        "cultivation_tips": "Requires minimal irrigation, tolerant to drought, benefits from proper seed treatment"
    },
    "muskmelon": {
        "name": "Muskmelon",
        "devnagri_name": "खरबूज",
        "image_url": "https://example.com/muskmelon.jpg",
        "description": "Muskmelon is a sweet, fragrant fruit belonging to the Cucurbitaceae family. It is characterized by its netted skin and sweet, juicy flesh. The plant is an annual vine that produces round to oval fruits with a distinctive aroma. Muskmelons are rich in vitamins A and C, and have high water content. They are consumed fresh, in salads, and as a refreshing summer fruit. Rajasthan and Maharashtra are significant muskmelon-producing states in India. The crop requires warm temperatures and well-drained soil for optimal growth.",
        "growing_season": "Summer (February-May)",
        "water_requirements": 600,
        "soil_requirements": "Sandy loam, well-drained soil (pH 6.0-7.0)",
        "common_varieties": ["Hara Madhu", "Pusa Sharbati", "Arka Rajhans"],
        "major_growing_regions": ["Rajasthan", "Maharashtra", "Gujarat"],
        "cultivation_tips": "Requires warm temperatures, controlled irrigation, and protection from frost"
    },
    "orange": {
        "name": "Orange",
        "devnagri_name": "संत्रा",
        "image_url": "https://example.com/orange.jpg",
        "description": "Orange is a citrus fruit known for its vibrant color and high vitamin C content. The tree is an evergreen that produces round fruits with segmented, juicy flesh. Oranges are consumed fresh, processed into juice, and used in various culinary and medicinal applications. Maharashtra and Madhya Pradesh are major orange-producing states in India. The crop requires specific climatic conditions, including moderate temperatures and well-drained soil. Oranges are economically important and contribute significantly to the agricultural sector.",
        "growing_season": "Winter (November-February)",
        "water_requirements": 1000,
        "soil_requirements": "Well-drained, slightly acidic soil (pH 6.0-6.5)",
        "common_varieties": ["Nagpur Mandarin", "Coorg Mandarin", "Darjeeling Mandarin"],
        "major_growing_regions": ["Maharashtra", "Madhya Pradesh", "Assam"],
        "cultivation_tips": "Requires consistent moisture, protection from frost, and regular pruning"
    },
    "papaya": {
        "name": "Papaya",
        "devnagri_name": "पपई",
        "image_url": "https://example.com/papaya.jpg",
        "description": "Papaya is a tropical fruit tree known for its large, sweet fruits with numerous black seeds. The plant is fast-growing and produces fruits throughout the year in suitable climates. Papayas are rich in vitamins A and C, and contain the enzyme papain, which aids digestion. They are consumed fresh, in salads, and used in various culinary preparations. Karnataka and Maharashtra are significant papaya-producing states in India. The crop is valued for its nutritional benefits and versatility in food processing.",
        "growing_season": "Year-round in tropical regions",
        "water_requirements": 1200,
        "soil_requirements": "Well-drained, fertile soil (pH 6.0-6.5)",
        "common_varieties": ["Coorg Honey Dew", "Red Lady", "Pusa Dwarf"],
        "major_growing_regions": ["Karnataka", "Maharashtra", "Andhra Pradesh"],
        "cultivation_tips": "Requires warm temperatures, consistent moisture, and protection from strong winds"
    },
    "pigeonpeas": {
        "name": "Pigeon Peas",
        "devnagri_name": "तूर",
        "image_url": "https://example.com/pigeonpeas.jpg",
        "description": "Pigeon peas are a leguminous crop widely cultivated in tropical and subtropical regions. The plant is a perennial shrub that produces edible seeds rich in protein, vitamins, and minerals. They are an important source of nutrition in Indian cuisine, used in dal and other traditional dishes. Pigeon peas are drought-resistant and improve soil fertility through nitrogen fixation. Maharashtra and Karnataka are major producing states. The crop plays a crucial role in sustainable agriculture and food security.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 650,
        "soil_requirements": "Well-drained, medium to black soil (pH 6.0-7.5)",
        "common_varieties": ["BSMR-853", "ICPL-87", "Asha"],
        "major_growing_regions": ["Maharashtra", "Karnataka", "Madhya Pradesh"],
        "cultivation_tips": "Tolerant to drought, requires minimal irrigation, benefits from intercropping"
    },
    "pomegranate": {
        "name": "Pomegranate",
        "devnagri_name": "डाळिंब",
        "image_url": "https://example.com/pomegranate.jpg",
        "description": "Pomegranate is a deciduous shrub or small tree producing large, red fruits filled with edible seeds. The fruit is known for its high antioxidant content and numerous health benefits. Maharashtra is the leading pomegranate-producing state in India. The crop is valued for fresh consumption, juice production, and export. Pomegranate trees are drought-resistant and can thrive in semi-arid conditions. The fruit is rich in vitamins, minerals, and has significant medicinal properties.",
        "growing_season": "Winter (October-February)",
        "water_requirements": 800,
        "soil_requirements": "Well-drained, deep soil (pH 6.5-7.5)",
        "common_varieties": ["Bhagwa", "Ganesh", "Arakta"],
        "major_growing_regions": ["Maharashtra", "Gujarat", "Karnataka"],
        "cultivation_tips": "Requires pruning, controlled irrigation, and protection from extreme temperatures"
    },
    "rice": {
        "name": "Rice",
        "devnagri_name": "तांदूळ",
        "image_url": "https://example.com/rice.jpg",
        "description": "Rice is a staple food crop belonging to the Poaceae family, cultivated in flooded fields across tropical and subtropical regions. It is the primary food source for over half the world's population. The plant is an annual grass that produces edible grains in clusters. India is the second-largest rice producer globally, with significant cultivation in states like West Bengal, Punjab, and Andhra Pradesh. Rice comes in multiple varieties, including long-grain, short-grain, and aromatic types like Basmati.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 1200,
        "soil_requirements": "Clay or clay loam soil (pH 6.0-7.0)",
        "common_varieties": ["Basmati", "IR64", "Sona Masuri"],
        "major_growing_regions": ["West Bengal", "Punjab", "Andhra Pradesh"],
        "cultivation_tips": "Requires flooded fields, consistent water management, and careful transplantation"
    },
    "watermelon": {
        "name": "Watermelon",
        "devnagri_name": "पाणतंबूज",
        "image_url": "https://example.com/watermelon.jpg",
        "description": "Watermelon is a flowering plant species in the Cucurbitaceae family, known for its large, sweet fruit with high water content. The plant is a vine-like annual with large leaves and produces massive fruits weighing up to 20 kg. Rich in vitamins, minerals, and lycopene, watermelons are popular summer fruits. India is a significant watermelon producer, with major cultivation in Rajasthan, Maharashtra, and Karnataka. The crop is valued for both fresh consumption and commercial production.",
        "growing_season": "Summer (March-July)",
        "water_requirements": 600,
        "soil_requirements": "Sandy loam, well-drained soil (pH 6.0-7.5)",
        "common_varieties": ["Sugar Baby", "Crimson Sweet", "Charleston Gray"],
        "major_growing_regions": ["Rajasthan", "Maharashtra", "Karnataka"],
        "cultivation_tips": "Requires warm temperatures, good drainage, and careful pollination"
    },
    "pigeonpeas": {
        "name": "Pigeon Peas",
        "devnagri_name": "तूर",
        "image_url": "https://example.com/pigeonpeas.jpg",
        "description": "Pigeon peas are a perennial legume crop belonging to the Fabaceae family. Also known as Arhar or Tur, these plants are crucial in Indian agriculture for their protein-rich seeds and soil-improving properties. The crop is drought-resistant and can grow in various soil conditions. Pigeon peas are extensively used in dal preparations and are an important source of protein in vegetarian diets. Maharashtra and Madhya Pradesh are major producing states.",
        "growing_season": "Kharif (June-September)",
        "water_requirements": 500,
        "soil_requirements": "Well-drained, medium-deep soil (pH 6.0-7.5)",
        "common_varieties": ["BSMR-736", "ICPL-87", "Asha"],
        "major_growing_regions": ["Maharashtra", "Madhya Pradesh", "Karnataka"],
        "cultivation_tips": "Drought-tolerant, requires minimal irrigation, benefits from intercropping"
    },
    "pomegranate": {
        "name": "Pomegranate",
        "devnagri_name": "डाळिंब",
        "image_url": "https://example.com/pomegranate.jpg",
        "description": "Pomegranate is a deciduous shrub or small tree in the Lythraceae family, known for its distinctive red fruits filled with edible seeds. The plant produces beautiful red flowers and large, round fruits with numerous seeds. Rich in antioxidants, vitamins, and minerals, pomegranates are valued for both culinary and medicinal purposes. Maharashtra is the leading pomegranate-producing state in India, with significant exports to global markets.",
        "growing_season": "Perennial (fruits in October-January)",
        "water_requirements": 800,
        "soil_requirements": "Well-drained, sandy loam soil (pH 5.5-7.5)",
        "common_varieties": ["Bhagwa", "Ganesh", "Arakta"],
        "major_growing_regions": ["Maharashtra", "Karnataka", "Gujarat"],
        "cultivation_tips": "Requires pruning, controlled irrigation, and protection from extreme temperatures"
    },
    "orange": {
        "name": "Orange",
        "devnagri_name": "संत्रा",
        "image_url": "https://example.com/orange.jpg",
        "description": "Orange is a citrus fruit belonging to the Rutaceae family, known for its sweet, juicy segments and high vitamin C content. The tree is evergreen, producing white flowers and round fruits. Nagpur in Maharashtra is famous for its distinctive orange varieties. Oranges are consumed fresh, processed into juice, and used in various culinary and medicinal applications. The crop requires specific climatic conditions and careful management.",
        "growing_season": "Winter (November-February)",
        "water_requirements": 900,
        "soil_requirements": "Well-drained, slightly acidic soil (pH 6.0-6.5)",
        "common_varieties": ["Nagpur Mandarin", "Kinnow", "Malta"],
        "major_growing_regions": ["Maharashtra", "Madhya Pradesh", "Assam"],
        "cultivation_tips": "Requires protection from frost, consistent moisture, and careful pruning"
    }

}

def get_crop_recommendation(item):
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_model.pkl')

    with open(scaler_path, 'rb') as f:
        crop_scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        crop_model = pickle.load(f)

    scaled_item = crop_scaler.transform(np.array(item).reshape(1, -1))
    prediction = crop_model.predict(scaled_item)  # Ensure it returns an array

    if isinstance(prediction, np.ndarray):
        prediction = prediction.item()  # Convert single-element array to scalar

    crop_name = crops[int(prediction)]
    
    # Return full crop details
    return crop_details.get(crop_name, {"name": crop_name, "description": "Details not available"})




def get_fertilizer_recommendation(num_features, cat_features):
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_model.pkl')
    
    with open(scaler_path, 'rb') as f:
        fertilizer_scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        fertilizer_model = pickle.load(f)

    scaled_features = fertilizer_scaler.transform(np.array(num_features).reshape(-1, len(num_features)))
    cat_features = np.array(cat_features).reshape(-1, len(cat_features))
    item = np.concatenate([scaled_features, cat_features], axis=1)
    prediction = fertilizer_model.predict(item)[0]
    return fertilizer_classes[prediction]

crop_diseases_classes = {'maize': [(0, 'Cercospora leaf spot Gray leaf spot'),
				 (1, 'Common rust'),
				 (2, 'Northern Leaf Blight'),
				 (3, 'Healthy')]}

crop_list = list(crop_diseases_classes.keys())


crops = {'apple': 1, 'banana': 2, 'blackgram': 3, 'chickpea': 4, 'coconut': 5, 'coffee': 6, 'cotton': 7, 'grapes': 8, 'jute': 9, 'kidneybeans': 10, 'lentil': 11, 'maize': 12, 'mango': 13, 'mothbeans': 14, 'mungbean': 15, 'muskmelon': 16, 'orange': 17, 'papaya': 18, 'pigeonpeas': 19, 'pomegranate': 20, 'rice': 21, 'watermelon': 22}

crops = list(crops.keys())

soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
Crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']

fertilizer_classes = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
