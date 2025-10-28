import time

def measure_latency(fn_to_execute, rounds=20):

    warmup = 5
    latencies = []

    # warmup rounds (cold start)
    for _ in range(warmup):
        fn_to_execute()

    # total rounds
    for _ in range(rounds):
        start = time.perf_counter()
        fn_to_execute()
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
    
    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    return avg_lat, max_lat
        

