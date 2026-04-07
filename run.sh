# 删除缓存
killall -9 python

#运行脚本

#工具测试脚本
# 终端 1
conda activate tool1
cd tool_server/tool_workers/scripts/launch_scripts
python start_server_local.py --config ./config/all_service_example_local.yaml

# 终端 2
conda activate tool1
python scripts/tools_test/tools_test.py 

# AddIndex
export https_proxy=http://10.31.90.11:7897 http_proxy=http://10.31.90.11:7897 all_proxy=socks5://10.31.90.11:7897