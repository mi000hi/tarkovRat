import requests

def run_query(query):
    response = requests.post('https://api.tarkov.dev/graphql', json={'query': query})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(response.status_code, query))


water_query = """
{
    items(name: "water") {
        name
        avg24hPrice
        changeLast48hPercent
        sellFor {
          price
          source
        }
    }
}
"""

# lists all items
query_all_items_prices = """
{
    items {
        name
        id
        width
        height
        avg24hPrice
        changeLast48hPercent
        sellFor {
            price
            source
        }
    }
}
"""

def getAllItemsPrices():
    result = run_query(query_all_items_prices)
    #result = run_query(water_query)
    return result['data']['items']
