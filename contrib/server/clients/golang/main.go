package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/google/gemma.cpp/contrib/server/clients/golang/pb"
)

func main() {
	err := run(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	serverAddress := "127.0.0.1:50051"
	flag.StringVar(&serverAddress, "addr", serverAddress, "server address")

	flag.Parse()
	var opts []grpc.DialOption

	opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))

	conn, err := grpc.Dial(serverAddress, opts...)
	if err != nil {
		return fmt.Errorf("dialing %q: %w", serverAddress, err)
	}
	defer conn.Close()
	client := pb.NewLLMClient(conn)

	stream, err := client.Converse(ctx)
	if err != nil {
		return fmt.Errorf("creating LLM request stream: %w", err)
	}

	for _, prompt := range []string{
		"Why is the sky blue?",
		"Can you explain it like I am five?",
	} {
		stream.Send(&pb.ConverseRequest{
			Text: prompt,
		})
		fmt.Printf("\n> %s\n", prompt)

		for {
			response, err := stream.Recv()
			if err != nil {
				return fmt.Errorf("error from LLM stream: %w", err)
			}

			if response.EndOfResponse {
				fmt.Printf("\n")
				break
			}

			for _, token := range response.Text {
				fmt.Print(token)
			}
		}
	}

	return nil
}
