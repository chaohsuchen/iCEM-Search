#!/usr/bin/env bash
# restart_srv.sh PORT [TOWN]: kill the CARLA server on PORT, start it again, wait until it answers, load TOWN.
PORT=$1; TOWN=${2:-}
alive() { timeout 20 ~/rcenv/bin/python -c "import carla; c=carla.Client('localhost',$PORT); c.set_timeout(10); c.get_world()" 2>/dev/null; }
pkill -f "[C]arlaUE4-Linux-Shipping.*carla-rpc-port=$PORT " ; sleep 5
pkill -9 -f "[C]arlaUE4-Linux-Shipping.*carla-rpc-port=$PORT " ; sleep 3
(cd ~/carla0915 && exec setsid nohup ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port=$PORT -carla-streaming-port=$((PORT + 1)) >> ~/carla_server_$PORT.log 2>&1 < /dev/null) > /dev/null 2>&1 &
for k in $(seq 1 60); do sleep 5; alive && break; done
alive || { echo "server $PORT did not come up"; exit 1; }
if [ -n "$TOWN" ]; then
  timeout 200 ~/rcenv/bin/python -c "
import carla, time
c = carla.Client('localhost', $PORT); c.set_timeout(180)
c.load_world('$TOWN')
for _ in range(60):
    time.sleep(2)
    if c.get_world().get_map().name.endswith('$TOWN'): break
print('$PORT', c.get_world().get_map().name)" 2>&1 | grep -v Warning | tail -1
fi
