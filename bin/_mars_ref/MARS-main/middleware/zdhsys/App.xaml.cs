﻿using System;
using System.Net;
using System.Threading.Tasks;
using System.Windows;
using System.Linq;
using zdhsys;
using Newtonsoft.Json;
using System.IO;
using System.Text;
using System.Windows.Shapes;
using Newtonsoft.Json.Linq;

namespace zdhsys
{
    public partial class App : Application
    {
        private HttpListener _listener;
        private Task _serverTask;
        internal ExperimentData experimentData { get; private set; }

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            
            // 启动HTTP服务器
            StartHttpServer();
        }

        private void StartHttpServer()
        {
            _listener = new HttpListener();
            _listener.Prefixes.Add("http://*:50000/");
            _listener.Start();

            _serverTask = Task.Run(() => HandleRequests());
        }

        private async Task HandleRequests()
        {
            while (_listener.IsListening)
            {
                var context = await _listener.GetContextAsync();
                ProcessRequest(context);
            }
        }

        private void ProcessRequest(HttpListenerContext context)
        {
            try
            {
                var request = context.Request;
                var response = context.Response;

                // 处理路由
                if (request.Url.AbsolutePath == "/" && request.HttpMethod == "GET")
                {
                    var responseString = "{\"message\": \"Welcome to AI robot platform API!\"}";
                    var buffer = System.Text.Encoding.UTF8.GetBytes(responseString);

                    response.ContentType = "application/json";
                    response.ContentLength64 = buffer.Length;
                    response.OutputStream.Write(buffer, 0, buffer.Length);
                }
                else if (request.Url.AbsolutePath == "/sendScheme2RobotPlatform" && request.HttpMethod == "POST")
                {
                    try
                    {
                        using (var reader = new System.IO.StreamReader(request.InputStream, request.ContentEncoding))
                        {
                            var json = reader.ReadToEnd();
                            Console.WriteLine("Received JSON data:");
                            Console.WriteLine(json);

                            // 更新实验数据
                            experimentData = JsonConvert.DeserializeObject<ExperimentData>(json);
                            Console.WriteLine("Experiment data updated:");
                            Console.WriteLine($"Task ID: {experimentData.TaskId}");
                            Console.WriteLine($"Experiment Name: {experimentData.ExperimentName}");

                            // 创建任务ID文件夹
                            string currentDirectory = Directory.GetCurrentDirectory();
                            string directoryPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(System.IO.Path.GetDirectoryName(currentDirectory)), "PLdata", experimentData.TaskId);
                            Directory.CreateDirectory(directoryPath);
                            string jsonFilePath = System.IO.Path.Combine(directoryPath, "scheme.json");

                            try
                            {
                                // 将 JSON 字符串解析为 JObject
                                var jsonObject = JObject.Parse(json);

                                // 将对象序列化为格式化的 JSON 字符串（缩进为 2）
                                string formattedJson = jsonObject.ToString(Formatting.Indented);

                                // 将格式化的 JSON 写入文件
                                File.WriteAllText(jsonFilePath, formattedJson);

                                Console.WriteLine($"JSON 已成功写入文件：{jsonFilePath}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"发生错误：{ex.Message}");
                            }

                            var responseString = "{\"status\": \"success\", \"message\": \"Experiment data updated\"}";
                            var buffer = System.Text.Encoding.UTF8.GetBytes(responseString);

                            response.ContentType = "application/json";
                            response.ContentLength64 = buffer.Length;
                            response.OutputStream.Write(buffer, 0, buffer.Length);
                        }
                    }
                    catch (Exception ex)
                    {
                        response.StatusCode = (int)HttpStatusCode.BadRequest;
                        var errorResponse = new { error = ex.Message };
                        var buffer = System.Text.Encoding.UTF8.GetBytes(Newtonsoft.Json.JsonConvert.SerializeObject(errorResponse));
                        response.OutputStream.Write(buffer, 0, buffer.Length);
                    }
                }
                else if (request.Url.AbsolutePath == "/getPLdata" && request.HttpMethod == "GET")
                {
                    try
                    {
                        // 获取taskid参数
                        string taskId = experimentData.TaskId;
                        string currentDirectory = Directory.GetCurrentDirectory();

                        // 如果 taskId 为空，则尝试从 PLdata 文件夹中获取最新的日期文件夹
                        if (string.IsNullOrEmpty(taskId))
                        {
                            string plDataPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(System.IO.Path.GetDirectoryName(currentDirectory)), "PLdata");

                            // 检查 PLdata 文件夹是否存在
                            if (!Directory.Exists(plDataPath))
                            {
                                Console.WriteLine("PLdata directory does not exist.");
                                return;
                            }

                            // 获取 PLdata 文件夹中的所有子文件夹
                            var directories = Directory.GetDirectories(plDataPath)
                                                       .Where(dir => DateTime.TryParse(System.IO.Path.GetFileName(dir), out _)) // 确保文件夹名是有效的日期格式
                                                       .Select(dir => new { Path = dir, Date = DateTime.Parse(System.IO.Path.GetFileName(dir)) })
                                                       .OrderByDescending(x => x.Date) // 按日期降序排序
                                                       .ToList();

                            if (directories.Any())
                            {
                                // 取最新日期的文件夹名称作为 taskId
                                taskId = System.IO.Path.GetFileName(directories.First().Path);
                                Console.WriteLine($"Latest taskid from folder: {taskId}");
                            }
                            else
                            {
                                Console.WriteLine("No valid date folders found in PLdata.");
                                return;
                            }
                        }

                        // 构建路径
                        string directoryPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(System.IO.Path.GetDirectoryName(currentDirectory)), "PLdata", taskId);
                        Console.WriteLine($"Constructed directory path: {directoryPath}");

                        // 检查目录是否存在
                        if (!Directory.Exists(directoryPath))
                        {
                            response.StatusCode = (int)HttpStatusCode.NotFound;
                            var errorResponse = new { error = "Task directory not found" };
                            var buffer = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(errorResponse));
                            response.OutputStream.Write(buffer, 0, buffer.Length);
                            return;
                        }

                        // 获取最新的txt文件
                        var files = Directory.GetFiles(directoryPath, "*.TXT")
                            .Select(f => new FileInfo(f))
                            .OrderByDescending(f => f.LastWriteTime)
                            .ToList();

                        if (files.Count == 0)
                        {
                            response.StatusCode = (int)HttpStatusCode.NotFound;
                            var errorResponse = new { error = "No TXT files found" };
                            var buffer = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(errorResponse));
                            response.OutputStream.Write(buffer, 0, buffer.Length);
                            return;
                        }

                        // 读取最新文件内容
                        string latestFileContent = File.ReadAllText(files[0].FullName);

                        // 发送响应
                        var responseBuffer = Encoding.UTF8.GetBytes(latestFileContent);
                        response.ContentType = "text/plain";
                        response.ContentLength64 = responseBuffer.Length;
                        response.OutputStream.Write(responseBuffer, 0, responseBuffer.Length);
                    }
                    catch (Exception ex)
                    {
                        response.StatusCode = (int)HttpStatusCode.InternalServerError;
                        var errorResponse = new { error = ex.Message };
                        var buffer = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(errorResponse));
                        response.OutputStream.Write(buffer, 0, buffer.Length);
                    }
                }
                else
                {
                    response.StatusCode = (int)HttpStatusCode.NotFound;
                }
            }
            finally
            {
                context.Response.OutputStream.Close();
            }
        }

        protected override void OnExit(ExitEventArgs e)
        {
            // 关闭HTTP服务器
            _listener?.Stop();
            _listener?.Close();
            base.OnExit(e);
        }
    }
}
