import requests
import asyncio
from autogen_ext.tools.mcp import SseMcpToolAdapter, SseServerParams
from constant import *


# Create server params for the remote MCP service
server_params = SseServerParams(
    url="https://mcpserver.mat-ai.cn/sse",
    headers={"Content-Type": "application/json"},
    timeout=30,  # Connection timeout in seconds
)


def hybird_retrieval_from_knowledge_base(
        query: str,
        topk: int = 5
        ) -> str:
    """
    Retrieval for knowledge from the knowledge base based on the specified query and returns the topk results.
    
    Parameters:
    query (str): The query for knowledge retrieval.
    topk (int): The number of top results to return, default is 3.
    
    Returns:
    str: The result of the knowledge retrieval in JSON format.
    """
    url = HYBRID_RAG_HOST_URI
    headers = {
        'Authorization': f'Bearer {HYBRID_RAG_HOST_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": {"topK": topk},
        "query": query,
        "response_mode": "blocking",
        "user": "tangger",
        "files": []
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 524:
        print("Server is not responding. Please try again later. Maybe GPU was down in the container.")
        return None

    try:
        result = response.json()
    except ValueError:
        return [{"error": "Response is not in JSON format"}]

    useful_results = []
    try:
        import json
        # answer = eval(result.get("answer", "[]"))
        answer = json.loads(result.get("answer", "[]"))
        for i, item in enumerate(answer):
            metadata = item.get("metadata", {})
            useful_info = {
                "index": "reference_" + str(i),
                "paper_title": item.get("title"),
                "para_content": item.get("content"),
                "sim_score": metadata.get("score")
            }
            useful_results.append(useful_info)

    except Exception as e:
        return [{"error": f"Error processing result: {e}", "status": "TERMINATE"}]
    if useful_results == []:
        useful_results = "NULL"
    return str(useful_results)


def online_retrieval_with_bing_and_playwright(
        query: str,
        ) -> str:
    """
    Retrieval for knowledge from the web based on the bing and playwright.
    
    Parameters:
    query (str): The query for arxiv and pubmed retrieval.
    
    Returns:
    str: The result of the paper retrieval in TEXT format.
    """
    url = HYBRID_RAG_HOST_URI

    headers = {
        'Authorization': f'Bearer {ONLINE_RAG_HOST_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": "tangger",
        "files": []
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 524:
        print("Server is not responding. Please try again later. Maybe GPU was down in the container.")
        return None

    try:
        result = response.json()['answer']
    except ValueError:
        return [{"error": "Response is not in JSON format"}]

    return result


def paper_retrieval_from_arxiv_and_pubmed(
        query: str,
        ) -> str:
    """
    Retrieval for knowledge from the knowledge base based on the specified query and returns the topk results.
    
    Parameters:
    query (str): The query for arxiv and pubmed retrieval.
    
    Returns:
    str: The result of the paper retrieval in TEXT format.
    """
    url = HYBRID_RAG_HOST_URI

    headers = {
        'Authorization': f'Bearer {PAPER_RAG_HOST_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": "tangger",
        "files": []
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 524:
        print("Server is not responding. Please try again later. Maybe GPU was down in the container.")
        return None

    try:
        result = response.json()['answer']
    except ValueError:
        return [{"error": "Response is not in JSON format"}]

    return result



def scheme_convert_to_json():
    return """
        转换合成方案时，必须严格遵守如下预定义的JSON格式，每个JSON结构的字段必须填充完整，即使有些字段留空：
        ```json
        {
            "TaskId": "",
            "ExperimentName": "",
            "Materials": [
                { "MaterialId": "", "Name": "", "Formula": "", "Amount": "", "Unit": "", "Purity": "", "State": ""}
                // ...可在此处添加更多material对象
            ],

            "Containers": [{
                "ContainerId": "", "Name": "", "Capacity": "", "Unit": "", "MaterialOfConstruction": "", "Shape": "", "HeatResistant": "", "PressureRating": "",
                // ...可在此处添加更多container对象
            ],

            "Equipments": [{
                "EquipmentId": "", "Name": "", 
                "Parameters": {
                    // 具体设备参数（例如 rpm 范围, 温度范围等）
                },
                }
                // ...可在此处添加更多equipment对象
            ],

            "RobotWorkflow": [{ 
                "Stepid": "", "Description": "",
                "Actions": [{
                    "ActionType": "",  // limited robot action: "pick_container""place_container""pick_container_with_material""place_container_into_equipment""remove_container_from_equipment"
                    "ContainerId": "",
                    "MaterialId": "",
                    "EquipmentId": "",
                    }
                    // ...可在此处添加更多子动作
                ],
                "Dependencies": [
                    // 若需要依赖之前的若干 step_id，可列在这里，如 ["1", "2"]
                ],
                "StepOutput": {
                    "ContainerId": "",
                    "Contents": [
                    {
                        "MaterialId": "",
                        "Amount": "",
                        "Unit": ""
                    }
                    // ...可在此处列出执行完本步骤后容器中的产物或状态
                    ]
                }
                }
                // ...可在此处添加更多step对象
            ]
            }
        ```

        ### JSON结构主要字段说明
        1. TaskId 类型: 字符串 说明: 任务的唯一标识符，用于区分不同的任务 限制: 必须唯一，不能重复
        2. Materials 类型: 数组 说明: 使用的材料列表，每个材料包含ID、名称、数量和单位
        3. Containers 类型: 数组 说明: 使用的容器列表，每个容器包含ID、类型、容量、单位和可选的附加参数
            限制: a.数组中的每个对象必须包含以下字段：a.1 "Type": 容器类型（如烧杯、锥形瓶、离心管等） a.2 "Capacity": 容器的容量 a.3 "Unit": 容量的单位 
        4. Equipments 类型: 数组 说明: 使用的设备列表，每个设备包含ID、名称和可选的参数 限制: 数组中的每个对象必须包含 "Name" 字段 "Parameters" 为可选，可根据设备实际需求进行配置（如搅拌速度、超声功率、温度范围等）
        5. Workflow 类型: 数组 说明: 包含所有步骤的列表 限制:每个步骤都是一个对象 顺序重要（一般按步骤顺序依次执行）
        6. StepId 类型: 整数 说明: 步骤的唯一标识符，用于区分不同的步骤 限制: 必须唯一，不能重复
        7. Actions 类型: 数组 说明: 包含该步骤中所有动作的列表 限制: a.每个动作都是一个对象 b.动作在数组中的顺序通常会影响执行顺序
        8. ActionId 类型: 字符串 说明: 动作的唯一标识符，用于区分同一步骤内的不同动作 限制: 在同一步骤内必须唯一
        9. ActionType 类型: 字符串 说明: 动作的类型，但此处特别强调仅限于机械臂可执行的动作 限制: 
            a.必须是以下预定义类型之一（对应机械臂操作）：
                a.1 "pick_container" （拿容器）
                a.2 "place_container" （放容器）
                a.3 "pick_container_with_material" （拿容器接材料/把材料加到容器里时的动作）
                a.4 "place_container_into_equipment" （将容器放进某设备）
                a.5 "remove_container_from_equipment" （从设备中取出容器）
            b. 诸如“搅拌”、“超声”、“离心”等动作不在此列，它们属于设备自身的潜在动作，不在机械臂的动作范围内
        10. Dependencies 类型: 数组 说明: 依赖的前一步骤的 step_id 列表 限制: 每个依赖项必须是有效的 step_id 当本步骤需要等待前面若干步骤完成后再执行时，可通过此字段进行控制
        11. StepOutput 类型: 字符串 说明: 步骤的输出标识符，用于后续步骤的输入或引用 限制: 标识符应唯一且有意义 可用来表示该步骤总体产出或容器中的新溶液名等
"""

def upload_to_s3(json_data: str):
    import json
    import re
    import subprocess
    import sys
    import tempfile
    import datetime

    def install_boto3():
        try:
            # 检查 boto3 是否已安装
            import boto3
            print("boto3 已安装。")
        except ImportError:
            # 如果未安装，动态安装 boto3
            print("正在安装 boto3...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
            print("boto3 安装完成。")

    def handle_minio_upload(file_path: str, file_name: str) -> str:
        """统一处理MinIO上传"""
        import boto3
        try:
            client = boto3.client(
                's3',
                endpoint_url=MINIO_SERVER,
                aws_access_key_id=MINIO_KEY_ID,
                aws_secret_access_key=MINIO_KEY
            )
            client.upload_file(file_path, MINIO_BUCKET, file_name, ExtraArgs={"ACL": "private"})
            
            # 生成预签名 URL
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': "MINIO_BUCKET", 'Key': file_name},
                ExpiresIn=36000
            )
            # return url.replace("http://100.85.52.31:9000" or "", "https://s3-api.siat-mic.com")
            return url
        
        except Exception as e:
            # print(e)
            return f"Error: {str(e)}, Request human/user intervention."
        
    install_boto3()
    # 去掉可能存在的 ```json 和 ``` 标记
    json_data_cleaned = re.sub(r'```json|```', '', json_data).strip()
    try:
        # 尝试解析清理后的JSON数据
        data = json.loads(json_data_cleaned)
        # 取得task id
        task_id = data['TaskId']
        # print("解析后的JSON数据:", data)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            try:
                json.dump(data, temp_file, indent=4, ensure_ascii=False)
                temp_file.flush()  # 确保数据写入文件
                file_name = f"robotExprimentScheme_{task_id}.json"
                url = handle_minio_upload(temp_file.name, file_name)
                return f"JSON Scheme has been uploaded to S3 storage. The unique URL is: {url}, please pass it to the robot platform."
            except Exception as e:
                # print(f"写入临时文件或上传文件时出错: {e}")
                return f"Error: {str(e)}, Request human/user intervention."
    except json.JSONDecodeError as e:
        # print(f"JSON解析错误: {e}")
        return f"Error: {str(e)}, Request human/user intervention."


def default_func():
    return "Approved. Proceed as planned!"

def generate_task_id():
    import datetime
    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化时间为字符串
    formatted_time = now.strftime("%Y%m%d%H%M%S")
    # 生成任务ID
    task_id = f"task_{formatted_time}"
    return task_id


def sendScheme2RobotWorkstation(task_id: str, scheme_url: str):
    
    def mol2mg(formula: str, source_unit: str, target_unit: str, value: float):
        """
        将mol转换为mg
        Args:
            formula: 化学式，如CsPb
            source_unit: 源单位 (mol或mmol)
            target_unit: 目标单位 (mg)
            value: 要转换的值
        """
        import requests
        from periodictable import formula as chem_formula
        
        # 检查单位是否有效
        if source_unit.lower() not in ['mol', 'mmol'] or target_unit.lower() != 'mg':
            return {"status": "error", "message": "Invalid units. Only mol/mmol to mg conversion supported"}
            
        try:
            # 解析化学式并计算摩尔质量
            compound = chem_formula(formula)
            molar_mass = compound.mass  # 获取化合物摩尔质量 (g/mol)
            
            # 转换计算
            if source_unit.lower() == 'mol':
                mg_value = value * molar_mass * 1000  # mol -> g -> mg
            elif source_unit.lower() == 'mmol':
                mg_value = value * molar_mass  # mmol -> g -> mg
                
            return {
                "status": "success",
                "formula": formula,
                "value": round(mg_value, 4),  # 保留4位小数
                "unit": "mg"
            }
        
        except ValueError as e:
            return {"status": "error", "message": f"Invalid chemical formula: {formula}. Error: {str(e)}"}

    # 首先检查task_id是否和scheme_url是否匹配
    if task_id not in scheme_url:
        return {"status": "error", "message": "task_id and scheme_url do not match, Request human/user intervention."}
    
    # 读取scheme_url的内容
    import requests
    try:
        response = requests.get(scheme_url)
        response.raise_for_status()
        scheme_content = response.text
        # 读取scheme_content的内容为JSON
        import json
        scheme_data = json.loads(scheme_content)
        robot_scheme = {}
        robot_scheme['TaskId'] = scheme_data['TaskId']
        robot_scheme['ExperimentName'] = scheme_data['ExperimentName']
        materials = []
        for mat in scheme_data['Materials']:
            materials.append({
                "MaterialId": mat['MaterialId'],
                "Name": mat['Name'],
                "Amount": mat['Amount'] if "mol" not in mat['Unit'] else mol2mg(mat['Formula'], mat['Unit'], 'mg', float(mat['Amount']))["value"],
                "Unit": mat['Unit'] if "mol" not in mat['Unit'] else "mg",
                "Purity": mat['Purity'],
                "State": mat['State']
            })
        robot_scheme['Materials'] = materials
        robot_scheme['Containers'] = scheme_data['Containers']
        robot_scheme['Equipments'] = scheme_data['Equipments']
        robot_scheme['RobotWorkflow'] = scheme_data['RobotWorkflow']
        # print(scheme_data)
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Error reading scheme_url: {e}"}
    
    import requests
    url = PLATFORM_HTTP_SERVER_URI
    data = {"status": "ok"}
    try:
        response = requests.post(url, json=robot_scheme)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending scheme to robot workstation: {e}")
        return None

def sendScheme2MobileRobot(task_id: str, scheme_url: str):
    
    def mol2mg(formula: str, source_unit: str, target_unit: str, value: float):
        """
        将mol转换为mg
        Args:
            formula: 化学式，如CsPb
            source_unit: 源单位 (mol或mmol)
            target_unit: 目标单位 (mg)
            value: 要转换的值
        """
        import requests
        from periodictable import formula as chem_formula
        
        # 检查单位是否有效
        if source_unit.lower() not in ['mol', 'mmol'] or target_unit.lower() != 'mg':
            return {"status": "error", "message": "Invalid units. Only mol/mmol to mg conversion supported"}
            
        try:
            # 解析化学式并计算摩尔质量
            compound = chem_formula(formula)
            molar_mass = compound.mass  # 获取化合物摩尔质量 (g/mol)
            
            # 转换计算
            if source_unit.lower() == 'mol':
                mg_value = value * molar_mass * 1000  # mol -> g -> mg
            elif source_unit.lower() == 'mmol':
                mg_value = value * molar_mass  # mmol -> g -> mg
                
            return {
                "status": "success",
                "formula": formula,
                "value": round(mg_value, 4),  # 保留4位小数
                "unit": "mg"
            }
        
        except ValueError as e:
            return {"status": "error", "message": f"Invalid chemical formula: {formula}. Error: {str(e)}"}

    # 首先检查task_id是否和scheme_url是否匹配
    if task_id not in scheme_url:
        return {"status": "error", "message": "task_id and scheme_url do not match, Request human/user intervention."}
    
    # 读取scheme_url的内容
    import requests
    try:
        response = requests.get(scheme_url)
        response.raise_for_status()
        scheme_content = response.text
        # 读取scheme_content的内容为JSON
        import json
        scheme_data = json.loads(scheme_content)
        robot_scheme = {}
        robot_scheme['TaskId'] = scheme_data['TaskId']
        robot_scheme['ExperimentName'] = scheme_data['ExperimentName']
        robot_scheme['Containers'] = scheme_data['Containers']
        robot_scheme['Equipments'] = scheme_data['Equipments']

        # print(scheme_data)
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Error reading scheme_url: {e}"}
    
    import requests
    url = MOBILE_ROBOT_HTTP_SERVER_URI
    data = {"status": "ok"}
    try:
        response = requests.post(url, json=robot_scheme)
        response.raise_for_status()
        return str(response.json()) + "\n" + "task_id: " + task_id + "\n" + "scheme_url: " + scheme_url + "\n"
    except requests.exceptions.RequestException as e:
        print(f"Error sending scheme to robot workstation: {e}")
        return None

def readPLdata():
    def list_to_markdown_table(data):
        # 拆分表头和数据行
        headers = data[0].split("\t")  # 假设列之间用制表符（\t）分隔
        rows = [line.split("\t") for line in data[1:]]

        # 创建表头部分
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        # 创建数据行部分
        data_rows = []
        for row in rows:
            data_rows.append("| " + " | ".join(row) + " |")

        # 合并所有部分
        markdown_table = "\n".join([header_row, separator_row] + data_rows)
        return markdown_table

    import requests
    url = PL_SERVER_URI
    try:
        response = requests.get(url)
        response.raise_for_status()
        scheme_content = response.text
        table = scheme_content.split("Data Points\r\n")[0].split("Peaks\r\n")[-1].split("\r\n")
        table = [t for t in table if t.strip()]
        md_table = list_to_markdown_table(table)
        max_pl = table[1].split('\t')[2]
        return md_table + "\n" + f"The max PL value in this expriment is {max_pl}"
    
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # print(sendScheme2RobotWorkstation(task_id="20250225173342", scheme_url="http://100.85.52.31:9000/temp/robotExprimentScheme_task_20250225173342.json?AWSAccessKeyId=9bUtQL1Gpo9JB6o3pSGr&Signature=bkfes1Nv%2BVzwiUj05p7uHk%2B0P9g%3D&Expires=1740512067"))
    # print(get_latest_exp_log())
    # print(readPLdata())
    # print(paper_retrieval_from_arxiv_and_pubmed("CsPbBr3"))
    print(hybird_retrieval_from_knowledge_base("How to synthesize graphene nanoplatelets?"))