from sklearn.feature_extraction.text import CountVectorizer

# Example recipe
recipe = ['2 tbsp extra virgin olive oil', '100g/3½oz cooked sweet potato, cut into small chunks', '175g/6oz (around 3) cooked new potatoes, cut into small chunks', '100g/3½oz (¼ head) broccoli, stem sliced and remainder cut into small florets', '½ red pepper, seeds removed, cut into small chunks', '1 leek, cut into chunks', 'handful (50g/1¾oz) frozen peas', '1 red or green chilli, finely chopped (deseed first if you like)', '100g/3½oz feta, roughly broken', '6 large free-range eggs', 'sea salt and freshly ground black pepper', 'green salad leaves, to serve', 'Heat the oil in a medium non-stick frying pan (ideally ovenproof) and gently fry the sweet potato, new potatoes, broccoli, pepper and leek for 5–7 minutes, or until the potatoes are lightly browned and the other vegetables are just tender, stirring regularly. Stir in the peas, chilli and feta.', 'Break the eggs into a bowl and add a good pinch of salt and lots of black pepper. Beat well using a large metal whisk. Pour the eggs into the pan and give it a little shake, so they run down between all the vegetables.', 'Cook the frittata  over a gentle heat for 5 minutes without stirring, or until the egg is almost set. Meanwhile, preheat the grill to high. Place the frittata under the hot grill for 3–4 minutes, or until set.', 'Loosen the sides of the frittata and slide onto a board. Cut into wedges and serve with a green leafy salad.']

# Create a CountVectorizer object
cv = CountVectorizer()

# Fit the vectorizer to the recipe text
cv.fit(recipe)

# Convert the recipe text into a matrix of word counts
count_matrix = cv.transform(recipe)

print(count_matrix.shape)  # (16, 89)


'''
        cv = CountVectorizer(stop_words="english", min_df=1)
        cv.fit(text_list)
        count_matrix = cv.transform(text_list).toarray()
        scaler = MinMaxScaler()
        normalized_count_matrix = scaler.fit_transform(count_matrix)
        return normalized_count_matrix






    def return_recipie(self,url):
        response = None
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

        if response is not None:
            soup = BeautifulSoup(response.content, 'html.parser')
            try:
                ingredients_list = soup.find('ul', class_='recipe-ingredients__list')
                ingredients = list(ingredients_list.find_all('li'))
            except AttributeError:
                ingredients = []
            final = []
            for i in ingredients:
                final.append(i.text.strip())
            method_list = soup.find('ol', class_='recipe-method__list')
            method = method_list.find_all('li')
            for i in method:
                final.append(i.text.strip())
            return final
        else:
            return []

    def return_countV(self, text_list):
        cv = CountVectorizer(stop_words="english", min_df=1)
        cv.fit(text_list)
        count_matrix = cv.transform(text_list).toarray()
        scaler = MinMaxScaler()
        normalized_count_matrix = scaler.fit_transform(count_matrix)
        return normalized_count_matrix

    def ann(self):
        ann_df = pd.DataFrame(columns=['title', 'rating_avg', 'data'])
        for idx, row in self.df.iterrows():
            data_val = self.return_recipie(row['recipe_url'])
            new_df = pd.DataFrame({'title': [str(row['title'])],
                                'rating_avg': [float(1 if row['rating_avg'] > 4.2 else -1)],
                                'data': [self.return_countV(data_val)]})
            new_df.reset_index(drop=True, inplace=True)  # reset the index to avoid alignment issues
            ann_df = pd.concat([ann_df, new_df])
        
        #print(ann_df)
        ann_df = ann_df.select_dtypes(include=[np.number])
        print("-----------------------")
        #print(ann_df)


        #new_df = df.loc[:, ['id', 'title', 'image_url', 'recipe_url', 'rating_avg', 'total_time', 'category', 'cuisine', 'ingredients']]

        X = self.df.iloc[:,0:6].values
        y = self.df.iloc[:,6].values
        model = Sequential()
        model.add(Dense(10, input_dim=6, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X, y, epochs=150, batch_size=10, verbose=0)
        # make class predictions with the model
        predictions = model.predict(X)
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy*100))
        # summarize the first 5 cases
        for i in range(5):
            print('predicted %d (actual %d)' % (predictions[i], y[i]))
        



'''








