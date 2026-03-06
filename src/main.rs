use axum::{routing::get, Router};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().json().init();

    let app = Router::new().route("/health", get(|| async { "ok" }));

    let addr = "0.0.0.0:8082";
    tracing::info!("embed-server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
