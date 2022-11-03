import copy
from gerrypy.data.load import *
from gerrypy.analyze.districts import *
from scipy.spatial.distance import pdist, squareform


def create_city_to_tract_mapping(state, state_cities):
    G = load_adjacency_graph(state)
    tracts = load_tract_shapes(state).to_crs(state_cities.crs)

    city_tracts = {}
    # Find tracts with intersection of municipal boundaries
    for city, row in state_cities.iterrows():
        try:  # If census urban areas dataset
            city_name, states = row.NAME10.split(',')
            state_list = states.strip().split('--')
            if state != state_list[0]:
                continue
        except AttributeError:  # Census places dataset
            city_name = row.NAME

        overlapping_tracts = tracts.loc[tracts.geometry.intersects(row.geometry)]
        overlap_ratio = overlapping_tracts.intersection(row.geometry).area / overlapping_tracts.area
        matched_tracts = list(overlapping_tracts[overlap_ratio > .5].index)
        if len(matched_tracts) < 2:
            continue
        city_tracts[city_name] = matched_tracts

    # Take largest connected component
    for city, tract_list in city_tracts.items():
        city_graph = G.subgraph(tract_list)
        city_components = list(nx.connected_components(city_graph))
        giant_component_ix = np.argmax(np.array(list(map(len, city_components))))
        city_tracts[city] = city_components[giant_component_ix]

    # Make sure no overlap
    overlap = {}
    for city1, city1_tracts in city_tracts.items():
        overlap[city1] = {}
        for city2, city2_tracts in city_tracts.items():
            overlap[city1][city2] = len(set(city1_tracts).intersection(set(city2_tracts)))
    assert ((pd.DataFrame(overlap) > 0).sum() > 1).sum() == 0
    return city_tracts


def reindex_tracts(n_tracts, city_mapping):
    old_tracts = list(np.arange(n_tracts))
    city_tracts = set().union(*list(city_mapping.values()))
    non_city_tracts = list(set(old_tracts) - city_tracts)
    tract_mapping = {new_ix: old_ix for new_ix, old_ix in enumerate(non_city_tracts)}
    offset = len(non_city_tracts)
    for city_ix, city_tracts in enumerate(list(city_mapping.values())):
        tract_mapping[offset + city_ix] = list(city_tracts)
    # create inverse mapping
    old_index_to_new_index = {}
    for new_ix, old_ix in tract_mapping.items():
        if isinstance(old_ix, list):
            for ix in old_ix:
                old_index_to_new_index[ix] = new_ix
        else:
            old_index_to_new_index[old_ix] = new_ix
    return tract_mapping, old_index_to_new_index


def create_merged_graph(G, old_to_new):
    new_adjacency_graph = nx.Graph()
    new_adjacency_graph.add_nodes_from(set(list(old_to_new.values())))
    new_adjacency_graph.add_edges_from([
        (old_to_new[n1], old_to_new[n2]) for n1, n2 in G.edges
    ])
    new_adjacency_graph.remove_edges_from(nx.selfloop_edges(new_adjacency_graph))
    return new_adjacency_graph


def create_merged_state_df(state_df, new_to_old):
    n_old_nodes = len(state_df)
    n_new_nodes = len(new_to_old)

    old_new_ix_matrix = np.zeros((n_old_nodes, n_new_nodes))
    for new_ix, old_ixs in new_to_old.items():
        old_new_ix_matrix[old_ixs, new_ix] = 1

    sum_metrics = state_df[['area', 'population']]
    average_metrics = state_df.drop(columns=['area', 'population', 'GEOID'])

    sum_metric_df = aggregate_sum_metrics(old_new_ix_matrix, sum_metrics)
    average_metric_df = aggregate_average_metrics(old_new_ix_matrix, average_metrics,
                                                  state_df.population.values + 1)
    return pd.concat([sum_metric_df, average_metric_df], axis=1)


def create_merged_tract_shapes(tract_gdf, old_to_new):
    tract_gdf['new_index'] = pd.Series(old_to_new)
    return tract_gdf.dissolve(by='new_index', aggfunc='sum')


def multi_district_cities(state_df, city_mapping, state, ideal_pop_threshold=.99):
    ideal_pop = state_df.population.values.sum() / constants.seats[state]['house']
    threshold = ideal_pop * ideal_pop_threshold
    return [city for city, tract_list in city_mapping.items()
            if (state_df.loc[tract_list].population.sum() > threshold)]


def find_bottleneck_node(g_coarsened, new_to_old):
    city_subset = {k: v for k, v in new_to_old.items() if isinstance(v, list)}
    for city_node, tract_list in sorted(city_subset.items(),
                                        key=lambda x: len(x[1]),
                                        reverse=True):
        test_graph = copy.deepcopy(g_coarsened)
        test_graph.remove_node(city_node)

        components = [c for c in sorted(nx.connected_components(test_graph),
                                        key=len, reverse=True)]
        # If no bottleneck node continue
        if len(components) == 1:
            continue

        # If bottleneck exists assume all nodes not in the largest
        # connected component are trapped by the bottleneck
        return city_node, [node for component in components[1:] for node in component]
    return None, None


def remediate_bottleneck(state_df, bottleneck_node, trapped_nodes, new_to_old, threshold):
    city_mapping = {k: v for k, v in new_to_old.items() if isinstance(v, list)}
    trapped_tracts = [tract for node in trapped_nodes for tract in
                      (new_to_old[node] if isinstance(new_to_old[node], list)
                       else [new_to_old[node]])]

    trapped_population = state_df.loc[trapped_tracts].population.sum()
    bottleneck_population = state_df.loc[new_to_old[bottleneck_node]].population.sum()
    if bottleneck_population > threshold:
        del city_mapping[bottleneck_node]
    elif (bottleneck_population + trapped_population) > threshold:
        del city_mapping[bottleneck_node]
    else:  # merge trapped nodes with bottleneck
        city_mapping[bottleneck_node] += trapped_tracts
        for node in trapped_nodes:
            if node in city_mapping:
                del city_mapping[node]
    return city_mapping


def create_feasible_city_aggregation(state, cities):
    G = load_adjacency_graph(state)
    state_df = load_state_df(state)
    ideal_population = state_df.population.sum() / constants.seats[state]['house']

    raw_mapping = create_city_to_tract_mapping(state, cities)

    # Remove cities which are too large
    to_filter = multi_district_cities(state_df, raw_mapping, state, ideal_pop_threshold=1)
    for city in to_filter:
        print(f'Filtered {city} from {state}')
        del raw_mapping[city]

    # Create city coarsened graph
    new_to_old, old_to_new = reindex_tracts(len(state_df), raw_mapping)
    g_coarsened = create_merged_graph(G, old_to_new)

    # Remediate cities that induce a bottleneck in the graph
    potential_bottlenecks = True
    while potential_bottlenecks:
        bottleneck_node, trapped_nodes = find_bottleneck_node(g_coarsened, new_to_old)
        if bottleneck_node is None:
            potential_bottlenecks = False
        else:
            remediated_mapping = remediate_bottleneck(state_df,
                                                      bottleneck_node, trapped_nodes,
                                                      new_to_old, ideal_population)
            new_to_old, old_to_new = reindex_tracts(len(state_df), remediated_mapping)
            g_coarsened = create_merged_graph(G, old_to_new)
    return new_to_old, old_to_new


def create_optimization_input_for_municipal_hard_constraint(state, cities, save_path):
    G = load_adjacency_graph(state)
    state_df = load_state_df(state)

    new_to_old, old_to_new = create_feasible_city_aggregation(state, cities)
    print(f'Merged {len(old_to_new)} tracts into {len(new_to_old)}')

    merged_adjacency_graph = create_merged_graph(G, old_to_new)
    merged_state_df = create_merged_state_df(state_df, new_to_old)

    os.makedirs(save_path, exist_ok=True)

    edge_dists = dict(nx.all_pairs_shortest_path_length(merged_adjacency_graph))
    centroids = merged_state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    merged_state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(merged_adjacency_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))
    pickle.dump(new_to_old, open(os.path.join(save_path, 'new_ix_to_old_ix.p'), 'wb'))
    pickle.dump(old_to_new, open(os.path.join(save_path, 'old_ix_to_new_ix.p'), 'wb'))


def preprocess():
    for state in constants.seats:
        places = load_census_places(state)
        state_cities = places.query("LSAD == '25'")
        save_path = os.path.join(constants.OPT_DATA_PATH,
                                 "municipal_hard_constraint",
                                 state)
        print(f'Starting creation of municipal hard constraint dataset for {state}')
        create_optimization_input_for_municipal_hard_constraint(state, state_cities, save_path)


if __name__ == '__main__':
    preprocess()
