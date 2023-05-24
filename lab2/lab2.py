from bs4 import BeautifulSoup
try:
    fhand = open(’example.html’)
    html = fhand.read()
    soup = BeautifulSoup(html, ’html.parser’)
    car_details = soup.find('ul', class_='listing-key-specs')
    split_details = list(vehicle_details.stripped_strings)
    print(split_details)
except:
    print(’File Not Found’)


#[’2005 (02 reg)’, ’Hatchback’, ’85,000 miles’, ’Manual’, ’1.5L’, ’123 bhp’, ’Petrol’]