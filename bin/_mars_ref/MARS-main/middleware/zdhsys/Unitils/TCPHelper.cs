using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace zdhsys.Unitils
{
    public class TCPHelper
    {
        public string ip = "127.0.0.1";
        public int port = 8888;
        private TcpClient tcp = new TcpClient();
        private NetworkStream stream;
        private Thread th = null;

        public bool Connect(string ip,int port)
        {
            try
            {
                // 连接服务器
                //await tcp.ConnectAsync(ip, port);
                tcp.Connect(ip,port);
                if (tcp.Connected)
                {
                    stream = tcp.GetStream();
                    th = new Thread(new ThreadStart(Recv));
                    th.Start();
                }
                return tcp.Connected;
            }
            catch { }
            return false;
        }

        public bool Connected()
        {
            try
            {
                if (tcp != null)
                {
                    return tcp.Connected;
                }
            }
            catch { }
            return false;
        }

        public void Send(string json)
        {
            // 异步发送数据
            byte[] sendData = Encoding.ASCII.GetBytes(json);
            stream.Write(sendData, 0, sendData.Length);
            Console.WriteLine("发送数据成功！");
        }

        public void Recv()
        {
            while (tcp.Connected)
            {
                try
                {
                    // 接收数据
                    byte[] receiveData = new byte[1024];
                    int bytesRead = stream.Read(receiveData, 0, receiveData.Length);
                    string receiveMessage = Encoding.ASCII.GetString(receiveData, 0, bytesRead);
                    Console.WriteLine("接收到服务器消息：{0}", receiveMessage);
                    Thread.Sleep(250);
                }
                catch {
                    Console.WriteLine("TCP断开连接");
                }
            }
        }

        public void Close()
        {
            try
            {
                tcp.Close();
            }
            catch { }
        }

    }
}
