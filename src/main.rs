use axum::{http::StatusCode, routing::get, Router};
use polars::prelude::*;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new().route("/", get(root));

    let listener = TcpListener::bind("0.0.0.0:8000")
        .await
        .expect("can start web server on port 8000");
    tracing::info!("listening on {:?}", listener.local_addr());
    axum::serve(listener, app)
        .await
        .expect("can start web server");
}

async fn root() -> Result<String, StatusCode> {
    Ok(String::from("Hello, World"))
}
